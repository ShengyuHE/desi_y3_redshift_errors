"""Submit the allocation commands in a launch manifest after successful dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


ACTIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
                 'REQUEUED', 'RESIZING', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SIGNALING'}


def dependency_states(job_ids):
    result = subprocess.run(
        ['sacct', '-j', ','.join(map(str, job_ids)), '--allocations', '--noheader',
         '--parsable2', '--format=JobIDRaw,State'], check=True,
        capture_output=True, text=True, timeout=30)
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[0].isdigit():
            states[int(fields[0])] = fields[1].split()[0].rstrip('+')
    return {job: states.get(job, 'UNKNOWN') for job in job_ids}


def main(filename):
    path = Path(filename).resolve()
    manifest = json.loads(path.read_text())
    if any(allocation.get('controller_pid') for allocation in manifest['allocations']):
        raise RuntimeError('Allocations already submitted; refusing duplicate launch.')

    def save(status):
        manifest['submission_status'] = status
        manifest['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(path)

    previous = None
    save('waiting_for_dependencies')
    while True:
        try:
            states = dependency_states(manifest['afterok'])
        except (subprocess.SubprocessError, OSError) as exc:
            print(f'Dependency query failed; retrying: {exc}', flush=True)
            time.sleep(30)
            continue
        if states != previous:
            print('Dependency states:', states, flush=True)
            previous = states
        manifest['dependency_states'] = states
        failed = {job: state for job, state in states.items()
                  if state not in ACTIVE_STATES | {'COMPLETED', 'UNKNOWN'}}
        if failed:
            save('dependency_failed')
            raise RuntimeError(f'Dependencies did not complete successfully: {failed}')
        if all(state == 'COMPLETED' for state in states.values()):
            break
        time.sleep(30)

    # Verify the prerequisite chains as well as the scheduler exit status.
    missing = [name for name in manifest['required_chain_files'] if not Path(name).is_file()]
    if missing:
        manifest['missing_dependency_outputs'] = missing
        save('dependency_outputs_missing')
        raise RuntimeError(f'{len(missing)} prerequisite chains are missing.')
    env = os.environ.copy()
    for key in ['SLURM_JOB_ID', 'SLURM_JOBID', 'SLURM_JOB_NUM_NODES', 'FIT_SCRIPT_DIR',
                'CORR_DV', 'JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'CUDA_VISIBLE_DEVICES']:
        env.pop(key, None)
    env.update(manifest['launch_environment'])
    for allocation in manifest['allocations']:
        with Path(allocation['log']).open('a') as log:
            process = subprocess.Popen(allocation['command'], cwd=manifest['repository'],
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, env=env, start_new_session=True)
        allocation['controller_pid'] = process.pid
        allocation['submitted_utc'] = datetime.now(timezone.utc).isoformat()
        save('submitting_allocations')
        print(f"Submitted allocation {allocation['group']}, controller PID {process.pid}", flush=True)
    save('allocations_submitted')


if __name__ == '__main__':
    main(sys.argv[1])
