#!/usr/bin/env python3
"""
CORRECTED: Zero-shot CLIP evaluation and baseline experiments with PROPER validation split
"""

import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json

class AFHQDataset(Dataset):
    """
    CORRECTED + ROBUST: Loads AFHQ validation set with proper paths and skips corrupted images
    """
    def __init__(self, real_root, fake_root, split='val', transform=None):
        self.samples = []
        self.transform = transform

        # Real images
        real_path = Path(real_root) / split
        print(f"Loading real images from: {real_path}")
        real_count = 0
        for animal_type in ['cat', 'dog', 'wild']:
            animal_dir = real_path / animal_type
            if animal_dir.exists():
                for img_file in animal_dir.glob('*.jpg'):
                    if self._verify_image(img_file):
                        self.samples.append((str(img_file), 0))
                        real_count += 1

        print(f"Loaded {real_count} real images")

        # Fake images
        fake_path = Path(fake_root)
        print(f"Loading fake images from: {fake_path}")
        fake_count = 0
        corrupted_count = 0

        for stylegan_dir in fake_path.iterdir():
            if stylegan_dir.is_dir() and 'afhqv2' in stylegan_dir.name:
                print(f"  Loading from: {stylegan_dir.name}")
                for img_file in stylegan_dir.glob('*.png'):
                    if self._verify_image(img_file):
                        self.samples.append((str(img_file), 1))
                        fake_count += 1
                    else:
                        corrupted_count += 1

        if corrupted_count > 0:
            print(f"⚠️  Skipped {corrupted_count} corrupted fake images")

        print(f"Loaded {fake_count} fake images")
        print(f"Total: {len(self.samples)} images for {split}")

    def _verify_image(self, img_path):
        """Verify that an image can be opened"""
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Robust image loading
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Failed to load {img_path}: {e}")
            image = Image.new('RGB', (512, 512), color='black')

        if self.transform:
            image = self.transform(image)
        return image, label

def zero_shot_eval():
    """
    Zero-shot CLIP evaluation on CORRECTED validation set
    """
    print("="*42)
    print("CORRECTED ZERO-SHOT CLIP EVALUATION")
    print("="*42)

    # CORRECTED PATHS
    REAL_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq'
    VAL_FAKE_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load CLIP
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(DEVICE)
    model.eval()

    # Text prompts
    texts = [
        "a real photograph",
        "an AI-generated synthetic image"
    ]
    print("\nText prompts:")
    for i, text in enumerate(texts):
        print(f"  [{i}] {text}")

    text_tokens = tokenizer(texts).to(DEVICE)

    # Load validation dataset (CORRECTED path)
    val_dataset = AFHQDataset(REAL_ROOT, VAL_FAKE_ROOT, split='val', transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    print(f"\nEvaluating on {len(val_dataset)} validation images...")

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)

            # Get image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity with text prompts
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Predict based on which text prompt has higher similarity
            fake_probs = similarity[:, 1].cpu().numpy()  # Probability of "fake"
            preds = (fake_probs > 0.5).astype(int)

            all_probs.extend(fake_probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*60)
    print("ZERO-SHOT CLIP RESULTS (CORRECTED)")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Real', 'Fake']))

    # Save results
    results = {
        'model': 'Zero-Shot CLIP (CORRECTED)',
        'accuracy': float(accuracy),
        'auc': float(auc)
    }

    with open('zero_shot_results_CORRECTED.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved results to zero_shot_results_CORRECTED.json")

def main():
    print("="*42)
    print("AFHQ Baseline Experiments (CORRECTED)")
    print("="*42)

    print("\n1. Running Zero-Shot CLIP evaluation...")
    print("-"*42)
    zero_shot_eval()

    print("\n" + "="*42)
    print("Baseline Experiments Complete!")
    print("="*42)

if __name__ == '__main__':
    main()