import zipfile
import os
import shutil
from pathlib import Path

docx_path = r"d:\SEDIC_2026\SEDIC_2026_Report_COMPLETED.docx"
assets_dir = r"d:\SEDIC_2026\assets"
os.makedirs(assets_dir, exist_ok=True)

if os.path.exists(docx_path):
    with zipfile.ZipFile(docx_path, 'r') as z:
        for filename in z.namelist():
            if filename.startswith('word/media/'):
                target_name = os.path.basename(filename)
                target_path = os.path.join(assets_dir, target_name)
                with z.open(filename) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted: {target_name} ({os.path.getsize(target_path)} bytes)")

# List extracted files
print("\nFiles in assets:")
for f in os.listdir(assets_dir):
    print(" -", f, os.path.getsize(os.path.join(assets_dir, f)), "bytes")
