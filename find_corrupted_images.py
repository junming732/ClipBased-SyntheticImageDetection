#!/usr/bin/env python3
"""
Find corrupted/truncated images in AFHQ dataset
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm

def check_image(img_path):
    """Try to open and verify an image"""
    try:
        img = Image.open(img_path)
        img.verify()  # Verify it's a valid image
        img = Image.open(img_path)  # Reopen after verify
        img.load()  # Actually load the image data
        return True, None
    except Exception as e:
        return False, str(e)

def scan_directory(root_path):
    """Scan directory for corrupted images"""
    print(f"\nScanning: {root_path}")

    corrupted = []
    total = 0

    for img_file in tqdm(list(Path(root_path).rglob('*.png')) + list(Path(root_path).rglob('*.jpg'))):
        total += 1
        is_valid, error = check_image(img_file)
        if not is_valid:
            corrupted.append((str(img_file), error))
            print(f"\n❌ Corrupted: {img_file}")
            print(f"   Error: {error}")

    print(f"\nScanned {total} images")
    print(f"Found {len(corrupted)} corrupted images")

    return corrupted

def main():
    print("="*60)
    print("Checking AFHQ Dataset for Corrupted Images")
    print("="*60)

    # Paths to check
    paths = [
        '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq/train',
        '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq/val',
        '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K',
        '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K'
    ]

    all_corrupted = []

    for path in paths:
        corrupted = scan_directory(path)
        all_corrupted.extend(corrupted)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total corrupted images: {len(all_corrupted)}")

    if all_corrupted:
        print("\nCorrupted files:")
        for img_path, error in all_corrupted:
            print(f"  {img_path}")

        # Save to file
        with open('corrupted_images.txt', 'w') as f:
            for img_path, error in all_corrupted:
                f.write(f"{img_path}\n")

        print("\n✓ Saved list to corrupted_images.txt")
        print("\nTo delete corrupted images, run:")
        print("  cat corrupted_images.txt | xargs rm")
    else:
        print("\n✓ No corrupted images found!")

if __name__ == '__main__':
    main()