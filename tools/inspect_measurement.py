import numpy as np
from pathlib import Path

data_dir = Path("data/zenodo")
sample_file = next(data_dir.glob("*.npy"))
raw_data = np.load(sample_file, allow_pickle=True)

print(f"Initial shape: {raw_data.shape}, dtype: {raw_data.dtype}\n")

for i in range(len(raw_data)):
    element = raw_data[i]
    if isinstance(element, np.ndarray):
        print(f"Index {i}: ndarray of shape {element.shape}, dtype {element.dtype}")
    elif isinstance(element, dict):
        print(f"Index {i}: dict with keys {list(element.keys())}")
    else:
        print(f"Index {i}: {type(element)} (Value: {element})")