"""
Create Kaggle-compatible zip with forward slashes
"""
import zipfile
import os
from pathlib import Path

# Source directory
source_dir = Path("data/raw/mudras/kaggle_50_mudras/images")
output_zip = "mudra_latest_dataset.zip"

print(f"Creating {output_zip}...")

# Create zip with forward slashes
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = Path(root) / file
            # Create archive name with forward slashes (relative to source_dir)
            arcname = file_path.relative_to(source_dir).as_posix()
            
            print(f"Adding: {arcname}")
            zipf.write(file_path, arcname=arcname)

print(f"\n✅ Created {output_zip}")
print(f"File size: {os.path.getsize(output_zip) / (1024*1024):.2f} MB")
print("\nVerifying paths in zip...")

# Verify paths use forward slashes
with zipfile.ZipFile(output_zip, 'r') as zipf:
    sample_names = zipf.namelist()[:5]
    print("\nSample paths (should use /):")
    for name in sample_names:
        print(f"  {name}")
    
    if any('\\' in name for name in zipf.namelist()):
        print("\n❌ WARNING: Backslashes found!")
    else:
        print("\n✅ All paths use forward slashes - Kaggle compatible!")
