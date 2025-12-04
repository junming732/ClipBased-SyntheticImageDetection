#!/usr/bin/env python3
"""
CORRECTED: Visualize misclassifications for both Fine-tuned CLIP and ResNet50
Uses proper validation dataset and handles corrupted images
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
from torchvision import models
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
        return image, label, img_path

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, feature_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classifier(features.float())
        return logits

def analyze_misclassifications(model, loader, device, model_name):
    """Find and analyze misclassified images"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    print(f"\nAnalyzing {model_name}...")

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Processing"):
            images = images.to(device, non_blocking=True)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Find misclassifications
    misclassified_idx = np.where(all_preds != all_labels)[0]

    # Separate into false positives and false negatives
    false_positives = []  # Real predicted as Fake
    false_negatives = []  # Fake predicted as Real

    for idx in misclassified_idx:
        item = {
            'path': all_paths[idx],
            'true_label': 'Real' if all_labels[idx] == 0 else 'Fake',
            'pred_label': 'Real' if all_preds[idx] == 0 else 'Fake',
            'confidence': all_probs[idx] if all_labels[idx] == 1 else 1 - all_probs[idx]
        }

        if all_labels[idx] == 0 and all_preds[idx] == 1:
            false_positives.append(item)
        elif all_labels[idx] == 1 and all_preds[idx] == 0:
            false_negatives.append(item)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total misclassifications: {len(misclassified_idx)} / {len(all_labels)}")
    print(f"False Positives (Real→Fake): {len(false_positives)}")
    print(f"False Negatives (Fake→Real): {len(false_negatives)}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Fake  {cm[1,0]:5d}  {cm[1,1]:5d}")

    return false_positives, false_negatives, all_probs, all_labels

def visualize_misclassifications(false_positives, false_negatives, model_name):
    """Visualize misclassified images"""

    # Visualize False Positives
    if false_positives:
        n_show = min(5, len(false_positives))
        # Sort by confidence (most confident mistakes first)
        fp_sorted = sorted(false_positives, key=lambda x: x['confidence'], reverse=True)[:n_show]

        fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
        fig.suptitle(f'{model_name}: False Positives (Real → Predicted as Fake)', fontsize=14)

        if n_show == 1:
            axes = [axes]

        for idx, item in enumerate(fp_sorted):
            img = Image.open(item['path']).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].set_title(f"Conf: {item['confidence']:.3f}", fontsize=11)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_false_positives.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {model_name.lower().replace(' ', '_')}_false_positives.png")
        plt.close()

    # Visualize False Negatives
    if false_negatives:
        n_show = min(5, len(false_negatives))
        fn_sorted = sorted(false_negatives, key=lambda x: x['confidence'], reverse=True)[:n_show]

        fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
        fig.suptitle(f'{model_name}: False Negatives (Fake → Predicted as Real)', fontsize=14)

        if n_show == 1:
            axes = [axes]

        for idx, item in enumerate(fn_sorted):
            img = Image.open(item['path']).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].set_title(f"Conf: {item['confidence']:.3f}", fontsize=11)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_false_negatives.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {model_name.lower().replace(' ', '_')}_false_negatives.png")
        plt.close()

def plot_confidence_distributions(clip_probs, clip_labels, resnet_probs, resnet_labels):
    """Plot confidence distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CLIP
    real_probs_clip = 1 - clip_probs[clip_labels == 0]
    fake_probs_clip = clip_probs[clip_labels == 1]

    axes[0].hist(real_probs_clip, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[0].hist(fake_probs_clip, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Confidence (Probability of being Real)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Fine-tuned CLIP Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ResNet
    real_probs_resnet = 1 - resnet_probs[resnet_labels == 0]
    fake_probs_resnet = resnet_probs[resnet_labels == 1]

    axes[1].hist(real_probs_resnet, bins=50, alpha=0.7, label='Real', color='blue', density=True)
    axes[1].hist(fake_probs_resnet, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_xlabel('Confidence (Probability of being Real)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('ResNet50 Confidence Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confidence_comparison.png")
    plt.close()

def main():
    # CORRECTED PATHS
    REAL_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq'
    VAL_FAKE_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K'

    CLIP_MODEL_PATH = 'best_afhq_clip_classifier_CORRECTED.pth'
    RESNET_MODEL_PATH = 'best_resnet_afhq_CORRECTED.pth'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("="*60)
    print("Analyzing Misclassifications (CORRECTED)")
    print("="*60)

    # ==================== Load Fine-tuned CLIP ====================
    print("\n1. Loading Fine-tuned CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    clip_model = clip_model.to(DEVICE)

    clip_classifier = CLIPClassifier(clip_model, feature_dim=512).to(DEVICE)
    checkpoint = torch.load(CLIP_MODEL_PATH, map_location=DEVICE)
    clip_classifier.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded CLIP from {CLIP_MODEL_PATH}")

    # Load CLIP validation dataset
    clip_dataset = AFHQDataset(REAL_ROOT, VAL_FAKE_ROOT, split='val', transform=clip_preprocess)
    clip_loader = DataLoader(clip_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Analyze CLIP
    clip_fp, clip_fn, clip_probs, clip_labels = analyze_misclassifications(
        clip_classifier, clip_loader, DEVICE, "Fine-tuned CLIP"
    )

    # ==================== Load ResNet50 ====================
    print("\n2. Loading ResNet50 model...")
    from torchvision import transforms

    resnet_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    resnet_model = models.resnet50(pretrained=False)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)
    resnet_model = resnet_model.to(DEVICE)

    checkpoint = torch.load(RESNET_MODEL_PATH, map_location=DEVICE)
    resnet_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded ResNet from {RESNET_MODEL_PATH}")

    # Load ResNet validation dataset
    resnet_dataset = AFHQDataset(REAL_ROOT, VAL_FAKE_ROOT, split='val', transform=resnet_transform)
    resnet_loader = DataLoader(resnet_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Analyze ResNet
    resnet_fp, resnet_fn, resnet_probs, resnet_labels = analyze_misclassifications(
        resnet_model, resnet_loader, DEVICE, "ResNet50"
    )

    # ==================== Visualizations ====================
    print("\n3. Creating visualizations...")

    # Visualize CLIP misclassifications
    if clip_fp or clip_fn:
        visualize_misclassifications(clip_fp, clip_fn, "Fine-tuned CLIP")

    # Visualize ResNet misclassifications
    if resnet_fp or resnet_fn:
        visualize_misclassifications(resnet_fp, resnet_fn, "ResNet50")

    # Plot confidence distributions
    plot_confidence_distributions(clip_probs, clip_labels, resnet_probs, resnet_labels)

    # Save detailed lists
    if clip_fp or clip_fn:
        with open('clip_misclassifications.txt', 'w') as f:
            f.write("Fine-tuned CLIP Misclassifications\n")
            f.write("="*60 + "\n\n")
            f.write(f"False Positives (Real → Fake): {len(clip_fp)}\n")
            for item in clip_fp:
                f.write(f"  {item['path']} (conf: {item['confidence']:.4f})\n")
            f.write(f"\nFalse Negatives (Fake → Real): {len(clip_fn)}\n")
            for item in clip_fn:
                f.write(f"  {item['path']} (conf: {item['confidence']:.4f})\n")
        print("✓ Saved clip_misclassifications.txt")

    if resnet_fp or resnet_fn:
        with open('resnet_misclassifications.txt', 'w') as f:
            f.write("ResNet50 Misclassifications\n")
            f.write("="*60 + "\n\n")
            f.write(f"False Positives (Real → Fake): {len(resnet_fp)}\n")
            for item in resnet_fp:
                f.write(f"  {item['path']} (conf: {item['confidence']:.4f})\n")
            f.write(f"\nFalse Negatives (Fake → Real): {len(resnet_fn)}\n")
            for item in resnet_fn:
                f.write(f"  {item['path']} (conf: {item['confidence']:.4f})\n")
        print("✓ Saved resnet_misclassifications.txt")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == '__main__':
    main()