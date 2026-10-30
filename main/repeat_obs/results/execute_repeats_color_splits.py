"""Execute and save the color-split notebook, including cell outputs on failure."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path(__file__).resolve().parents[1] / 'notebooks/repeats_color_splits.ipynb'
notebook = nbformat.read(path, as_version=4)

def progress(cell, cell_index, **kwargs):
    print(f'Executing cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

client = NotebookClient(notebook, timeout=1500, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f'COMPLETE: {path}', flush=True)
