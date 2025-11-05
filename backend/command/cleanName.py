import os
import glob

root_dir = "backend/app/data/raw/files"

count = 0
for subj in sorted(os.listdir(root_dir)):
    subj_dir = os.path.join(root_dir, subj)
    if not os.path.isdir(subj_dir):
        continue

    # Tìm mọi file kết thúc bằng "_named.edf"
    pattern = os.path.join(subj_dir, "*_named.edf")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
            count += 1
            print(f"[DEL] {file_path}")
        except Exception as e:
            print(f"[ERR] {file_path}: {e}")

print(f"\n✅ Done — deleted {count} files ending with '_named.edf'")
