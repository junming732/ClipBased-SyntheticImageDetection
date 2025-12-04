# Quick test script - save as test_dataset.py
from pathlib import Path

REAL_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq'
TRAIN_FAKE = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'
VAL_FAKE = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K'

print("=== TRAINING FAKE ===")
train_fake_files = set()
for d in Path(TRAIN_FAKE).iterdir():
    if d.is_dir():
        files = list(d.glob('*.png'))
        train_fake_files.update([f.name for f in files])
        print(f"{d.name}: {len(files)} files")

print(f"\nTotal unique train fake filenames: {len(train_fake_files)}")

print("\n=== VALIDATION FAKE ===")
val_fake_files = set()
for d in Path(VAL_FAKE).iterdir():
    if d.is_dir():
        files = list(d.glob('*.png'))
        val_fake_files.update([f.name for f in files])
        print(f"{d.name}: {len(files)} files")

print(f"\nTotal unique val fake filenames: {len(val_fake_files)}")

print("\n=== OVERLAP CHECK ===")
overlap = train_fake_files & val_fake_files
print(f"Overlapping filenames: {len(overlap)}")
if overlap:
    print("First 10 overlaps:", list(overlap)[:10])