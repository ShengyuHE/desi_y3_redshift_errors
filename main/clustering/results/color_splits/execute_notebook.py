"""Execute the color-split comparison notebook in the cosmodesi main environment."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path(__file__).resolve().parents[2] / 'notebooks/compare_color_splits.ipynb'
notebook = nbformat.read(path, as_version=4)
def progress(cell, cell_index, **kwargs):
    print(f'Cell {cell_index}: {cell.source.splitlines()[0]}', flush=True)
client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(path.parent)}}, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('COMPLETE:', path, flush=True)
