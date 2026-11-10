"""Minuit safeguards for likelihoods that call native numerical libraries."""

import time

import numpy as np
from desilike.profilers import Minuit


def _run_minuit(kernel, state, chi2, grad=None, max_iterations=int(1e5),
                phase_callback=lambda phase: None, **kwargs):
    """Follow desilike's Minuit kernel while exposing its phase and FMin status."""
    import iminuit

    options = {}
    if grad is not None:
        options['grad'] = lambda *values: np.asarray(grad(np.asarray(values)))
    minimizer = iminuit.Minuit(lambda *values: chi2(np.asarray(values)),
                               *state.start, **options)
    minimizer.errordef = 1.
    minimizer.strategy = 0 if state.fast else 1
    if kernel.precision is not None:
        minimizer.precision = float(kernel.precision)
    for index, ((low, high), proposal) in enumerate(zip(state.bounds, state.proposals)):
        minimizer.limits[index] = (None if np.isinf(low) else low,
                                   None if np.isinf(high) else high)
        if np.isfinite(proposal):
            minimizer.errors[index] = proposal

    def report(phase):
        fmin = minimizer.fmin
        kernel.last_diagnostics = dict(
            phase=phase, valid=bool(fmin.is_valid), edm=float(fmin.edm),
            edm_goal=float(fmin.edm_goal), nfcn=int(minimizer.nfcn),
            call_limit=bool(fmin.has_reached_call_limit),
            above_max_edm=bool(fmin.is_above_max_edm),
            accurate_covariance=bool(fmin.has_accurate_covar))
        log = kernel.logger.info if fmin.is_valid else kernel.logger.warning
        log('Minuit rank %d %s finished: valid=%s, calls=%d, EDM=%.6g '
            '(goal=%.6g), call_limit=%s, accurate_covariance=%s',
            kernel._rank, phase, fmin.is_valid, minimizer.nfcn, fmin.edm,
            fmin.edm_goal, fmin.has_reached_call_limit, fmin.has_accurate_covar)

    phase_callback('MIGRAD')
    try:
        minimizer.migrad(ncall=max_iterations)
    except RuntimeError as exc:
        kernel.logger.warning('Minuit rank %d MIGRAD failed: %r', kernel._rank, exc)
        return state
    report('MIGRAD')
    if not state.fast:
        phase_callback('HESSE')
        try:
            minimizer.hesse()
        except RuntimeError as exc:
            kernel.logger.warning('Minuit rank %d HESSE failed: %r', kernel._rank, exc)
        report('HESSE')
    state.logpdf = np.array(-.5 * minimizer.fval)
    state.best = np.array(list(minimizer.values))
    if not state.fast and minimizer.covariance is not None:
        state.cov = np.asarray(minimizer.covariance)
    return state


class FiniteMinuit(Minuit):
    """Reject invalid optimizer proposals before they reach the likelihood.

    COMET can return a non-finite likelihood outside its emulator domain. Minuit
    may subsequently propose NaN coordinates; passing these through a JAX
    callback into CLASS can segfault instead of raising a Python exception.
    The check must happen in Python before invoking the compiled objective.
    """

    def run(self, state, chi2, grad=None, **kwargs):
        from mpi4py import MPI

        self._rank = MPI.COMM_WORLD.rank
        self.last_diagnostics = None
        rejected = [0, 0]
        started = last_report = last_improvement = time.monotonic()
        calls, best = 0, np.inf
        phase = 'MIGRAD'

        def set_phase(name):
            nonlocal phase
            phase = name
            self.logger.info('Minuit rank %d starting %s; elapsed=%.1fs, calls=%d',
                             self._rank, phase, time.monotonic() - started, calls)

        def checked_chi2(values):
            nonlocal calls, best, last_report, last_improvement
            calls += 1
            if not np.isfinite(values).all():
                rejected[0] += 1
                value = np.inf
            else:
                value = float(chi2(values))
                if not np.isfinite(value):
                    rejected[1] += 1
                    value = np.inf
            now = time.monotonic()
            if value < best:
                best, last_improvement = value, now
            # Report on completed evaluations; this does not interrupt a native call.
            if now - last_report >= 60.:
                self.logger.info(
                    'Minuit rank %d %s progress: calls=%d, elapsed=%.1fs, '
                    'best_objective=%.8g, seconds_since_best=%.1f, '
                    'invalid_parameters=%d, invalid_objectives=%d',
                    self._rank, phase, calls, now - started, best,
                    now - last_improvement, *rejected)
                last_report = now
            return value

        try:
            return _run_minuit(self, state, checked_chi2, grad=grad,
                               phase_callback=set_phase, **kwargs)
        finally:
            self.logger.info('Minuit rank %d run ended: elapsed=%.1fs, calls=%d, '
                             'best_objective=%.8g', self._rank,
                             time.monotonic() - started, calls, best)
            if any(rejected):
                self.logger.warning(
                    'Minuit rank %d rejected %d non-finite parameter proposals before '
                    'likelihood evaluation and %d non-finite objective values.',
                    self._rank, *rejected)
