import hashlib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
from config import DATA_DIR, BASE_DIR

def generate_checksums():
    raw_files = sorted(list(DATA_DIR.glob("*.npy")))
    checksums = []
    
    print(f"Hashing {len(raw_files)} files...")
    
    for file_path in raw_files:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to manage memory efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
                
        checksums.append(f"{sha256_hash.hexdigest()}  {file_path.name}")
        
    out_path = BASE_DIR / "data" / "checksums.sha256"
    with open(out_path, "w") as f:
        f.write("\n".join(checksums) + "\n")
        
    print(f"Checksums successfully saved to {out_path.name}")

if __name__ == "__main__":
    generate_checksums()