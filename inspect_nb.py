import json
with open('c:/Users/basan/Desktop/project/Minor_Project/Fake_certificate_detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('c:/Users/basan/Desktop/project/Minor_Project/inspect_out.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            out_f.write(f"\n--- Cell {i+1} ---\n")
            out_f.write("".join(cell.get('source', [])))
            out_f.write("\n")
            if 'outputs' in cell and cell['outputs']:
                for out in cell['outputs']:
                    if out.get('output_type') == 'error':
                        out_f.write("\nERROR TRACEBACK:\n")
                        out_f.write("".join(out.get('traceback', [])))
                        out_f.write("\n")
