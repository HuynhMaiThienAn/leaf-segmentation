import json

notebook_path = "segmentation-test.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Metadata keys:", nb['metadata'].keys())
print("Metadata details:", json.dumps(nb['metadata'], indent=2))
print("Format:", nb['nbformat'], ".", nb['nbformat_minor'])
