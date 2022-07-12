from glob import glob

# Collect a list of all notebooks in the content folder
notebooks = glob("./*.ipynb", recursive=True)
python_files = glob("./*.py", recursive=True)
python_files.remove('./extract_workload.py')

with open('pack_workspace.py', 'a') as f:
    for py_file in python_files:
        with open(py_file,'r') as rf:
            for line in rf:
                f.write(line)
            f.write('\n')
    for ipath in notebooks:
        for cell in ntbk.cells:
            cell_tags = cell.get('metadata', {}).get('tags', [])
            if 'secure_api' in cell_tags:
                f.write(cell.get('source', {}))
                f.write('\n')
f.close()