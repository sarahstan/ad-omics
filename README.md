Welcome to the ADOmicsPy package!

Package can be installed using: uv install ./dist/adomicspy-0.1.0-py3-none-any.whl

Use thusly:
1. Import classes 
from adomicspy import scDATA, scPLOT
2. Import functions
from adomicspy.notebook import load_and_preprocess

Then, explore data:
1. Create instances of classes
loader = scDATA()
data = scDATA.load("path/to/data.csv")
2. Use combined functions
fig, data = load_and_visualize("path/to/data.csv")

Enjoy!

# Update

Install with 
```bash
uv init
uv pip install -e .
```

Import data like this:
```python
from data import scDATA
data = scDATA.load("path/to/data/folder")
scd = scDATA(data_path=data_path)
```