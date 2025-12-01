import pandas as pd
import os
from pathlib import Path

def create_cifake_csv(cifake_root, output_csv, split='train'):
    """
    Create CSV file for CIFAKE dataset compatible with grip-unina format

    Args:
        cifake_root: Path to CIFAKE dataset root
        output_csv: Output CSV filename
        split: 'train' or 'test'
    """
    data = []
    cifake_path = Path(os.path.expanduser(cifake_root))

    # Real images
    real_dir = cifake_path / split / 'REAL'
    print(f"Scanning {real_dir}...")
    for img_path in sorted(real_dir.glob('*.jpg')):
        data.append({
            'filename': str(img_path.absolute()),
            'typ': 'real'
        })

    # Fake images (AI-generated)
    fake_dir = cifake_path / split / 'FAKE'
    print(f"Scanning {fake_dir}...")
    for img_path in sorted(fake_dir.glob('*.jpg')):
        data.append({
            'filename': str(img_path.absolute()),
            'typ': 'fake'
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"\nCreated {output_csv}")
    print(f"Total images: {len(df)}")
    print(f"Real: {len(df[df['typ']=='real'])}")
    print(f"Fake: {len(df[df['typ']=='fake'])}")
    print(f"\nSample rows:")
    print(df.head())

    return df

if __name__ == '__main__':
    # Get dataset path from environment or use default
    CIFAKE_PATH = os.environ.get('CIFAKE_DATA_PATH', '~/nobackup_junming/CIFAKE')

    # Create CSVs for train and test splits
    print("="*60)
    print("Creating CIFAKE CSV files...")
    print("="*60)

    train_df = create_cifake_csv(CIFAKE_PATH, 'cifake_train.csv', 'train')
    print("\n" + "="*60 + "\n")
    test_df = create_cifake_csv(CIFAKE_PATH, 'cifake_test.csv', 'test')

    print("\n" + "="*60)
    print("Done! CSV files ready for use.")
    print("="*60)