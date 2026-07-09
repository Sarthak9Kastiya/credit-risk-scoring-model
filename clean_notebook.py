import json
import sys

def clean_notebook(filepath):
    with open(filepath, 'r') as f:
        nb = json.load(f)
        
    for cell in nb.get('cells', []):
        # Clear all outputs so no traces of the column remain
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
            
        # Clean the source code
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                # Remove the drop line
                if "df.drop('historical_default'" in line or "df=df.drop('historical_default'" in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source
            
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook cleaned successfully!")

if __name__ == '__main__':
    clean_notebook(sys.argv[1])
