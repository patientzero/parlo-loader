"""Execute a temporary notebook copy; never save outputs into the shipped notebook."""
import argparse
import os
from pathlib import Path
import re
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def verify(root):
    source = Path(__file__).with_name('explore_parlo.ipynb')
    original = source.read_bytes()
    nb = nbformat.read(source, as_version=4)
    notebook_code_cells = sum(cell.cell_type == 'code' for cell in nb.cells)
    assert all(not cell.get('outputs') and cell.get('execution_count') is None for cell in nb.cells if cell.cell_type == 'code')
    nb.cells[1].source = re.sub(r'^CORPUS_ROOT =.*$', lambda _: f'CORPUS_ROOT = Path({str(Path(root).resolve())!r})', nb.cells[1].source, flags=re.M)
    nb.cells.append(nbformat.v4.new_code_cell('''
# Exercise selector callbacks in this temporary verification copy.
task_selector.value = 'reading'
group_selector.value = 'HC'
update_participants()
assert len(participant_selector.options) > 1
participant_selector.value = participant_selector.options[1]
show_selection()
selected = selected_view().recordings.collect()
assert set(selected['task']) == {'reading'}
assert set(selected['diagnosis']) == {'HC'}
assert selected['participant_id'].n_unique() == 1
task_selector.value = 'pataka'
update_participants()
assert play_button.disabled
'''))
    with tempfile.TemporaryDirectory(prefix='parlo-notebook-') as temp:
        env = dict(os.environ, MPLCONFIGDIR=temp, IPYTHONDIR=temp, JUPYTER_RUNTIME_DIR=temp)
        km = KernelManager(kernel_name='python3')
        km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(nb, km=km, timeout=180, resources={'metadata': {'path': str(source.parent)}})
        try:
            client.execute(env=env)
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        executed = sum(cell.cell_type == 'code' and cell.execution_count is not None for cell in nb.cells)
        assert source.read_bytes() == original
        assert not any(output.get('output_type') == 'error' for cell in nb.cells for output in cell.get('outputs', []))
        print(f'Executed {notebook_code_cells} notebook code cells and {executed - notebook_code_cells} selector check successfully; shipped notebook remains cleared.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus_root')
    verify(parser.parse_args().corpus_root)
