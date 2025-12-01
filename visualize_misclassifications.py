#!/usr/bin/env python3
"""
Visualize Misclassified Images
Shows which images the AFHQ CLIP classifier got wrong
"""
import torch
from torch.utils.data import DataLoader
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_misclassifications(model_path='best_afhq_clip_classifier.pth', max_display=20):
    """Find and visualize misclassified images"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    print(f"Using device: {DEVICE}")

    # Load CLIP model
    print("Loading CLIP model...")
    model_name = 'ViT-B-32'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai'
    )
    clip_model = clip_model.to(DEVICE)

    # Load classifier
    from train_afhq_clip import CLIPClassifier, AFHQDataset
    classifier = CLIPClassifier(clip_model, feature_dim=512).to(DEVICE)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: {model_path} not found!")
        return

    classifier.eval()

    # Load validation dataset
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=preprocess)

    print(f"\nAnalyzing {len(val_dataset)} validation images...")

    # Find misclassifications
    misclassified = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            img_tensor, true_label = val_dataset[idx]
            img_path = val_dataset.samples[idx][0]

            # Get prediction
            output = classifier(img_tensor.unsqueeze(0).to(DEVICE)).squeeze()
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob > 0.5 else 0

            # Check if misclassified
            if pred_label != true_label:
                misclassified.append({
                    'path': img_path,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': prob if pred_label == 1 else (1 - prob)
                })

    print(f"\nFound {len(misclassified)} misclassifications out of {len(val_dataset)} images")
    print(f"Accuracy: {(len(val_dataset) - len(misclassified)) / len(val_dataset):.4f}")

    if len(misclassified) == 0:
        print("\n🎉 Perfect classification! No errors to visualize.")
        return

    # Separate by type
    false_positives = [m for m in misclassified if m['true_label'] == 0 and m['pred_label'] == 1]
    false_negatives = [m for m in misclassified if m['true_label'] == 1 and m['pred_label'] == 0]

    print(f"\nFalse Positives (Real → Predicted Fake): {len(false_positives)}")
    print(f"False Negatives (Fake → Predicted Real): {len(false_negatives)}")

    # Save detailed list
    with open('misclassified_images.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MISCLASSIFIED IMAGES\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total misclassifications: {len(misclassified)} / {len(val_dataset)}\n")
        f.write(f"Accuracy: {(len(val_dataset) - len(misclassified)) / len(val_dataset):.4f}\n\n")

        f.write("-"*80 + "\n")
        f.write(f"FALSE POSITIVES: {len(false_positives)} (Real images predicted as Fake)\n")
        f.write("-"*80 + "\n")
        for m in false_positives:
            f.write(f"Path: {m['path']}\n")
            f.write(f"Confidence: {m['confidence']:.4f}\n\n")

        f.write("-"*80 + "\n")
        f.write(f"FALSE NEGATIVES: {len(false_negatives)} (Fake images predicted as Real)\n")
        f.write("-"*80 + "\n")
        for m in false_negatives:
            f.write(f"Path: {m['path']}\n")
            f.write(f"Confidence: {m['confidence']:.4f}\n\n")

    print("✓ Saved detailed list to misclassified_images.txt")

    # Visualize false positives
    if len(false_positives) > 0:
        n_display = min(max_display, len(false_positives))
        n_cols = 5
        n_rows = (n_display + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, m in enumerate(false_positives[:n_display]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = Image.open(m['path']).convert('RGB')
            ax.imshow(img)
            ax.set_title(f"Real → Fake\nConf: {m['confidence']:.2f}", fontsize=9)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(n_display, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f'False Positives: Real Images Predicted as Fake (showing {n_display}/{len(false_positives)})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('false_positives.png', dpi=300, bbox_inches='tight')
        print("✓ Saved false_positives.png")
        plt.close()

    # Visualize false negatives
    if len(false_negatives) > 0:
        n_display = min(max_display, len(false_negatives))
        n_cols = 5
        n_rows = (n_display + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, m in enumerate(false_negatives[:n_display]):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            img = Image.open(m['path']).convert('RGB')
            ax.imshow(img)
            ax.set_title(f"Fake → Real\nConf: {m['confidence']:.2f}", fontsize=9)
            ax.axis('off')

        # Hide unused subplots
        for idx in range(n_display, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f'False Negatives: Fake Images Predicted as Real (showing {n_display}/{len(false_negatives)})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('false_negatives.png', dpi=300, bbox_inches='tight')
        print("✓ Saved false_negatives.png")
        plt.close()

def analyze_confidence_distribution(model_path='best_afhq_clip_classifier.pth'):
    """Analyze prediction confidence distribution"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    print("Loading model...")
    model_name = 'ViT-B-32'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai'
    )
    clip_model = clip_model.to(DEVICE)

    from train_afhq_clip import CLIPClassifier, AFHQDataset
    classifier = CLIPClassifier(clip_model, feature_dim=512).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # Load validation dataset
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print("Computing predictions...")

    real_probs = []
    fake_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = classifier(images.to(DEVICE)).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            for prob, label in zip(probs, labels):
                if label == 0:  # Real
                    real_probs.append(prob)
                else:  # Fake
                    fake_probs.append(prob)

    # Plot confidence distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(real_probs, bins=50, alpha=0.7, label='Real Images', color='blue', edgecolor='black')
    ax1.hist(fake_probs, bins=50, alpha=0.7, label='Fake Images', color='red', edgecolor='black')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax1.set_xlabel('Predicted Probability (Fake)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot([real_probs, fake_probs], labels=['Real Images', 'Fake Images'])
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    ax2.set_ylabel('Predicted Probability (Fake)', fontsize=12)
    ax2.set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved confidence_distribution.png")
    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("CONFIDENCE STATISTICS")
    print("="*60)
    print(f"\nReal Images (should be < 0.5):")
    print(f"  Mean: {np.mean(real_probs):.4f}")
    print(f"  Median: {np.median(real_probs):.4f}")
    print(f"  Min: {np.min(real_probs):.4f}")
    print(f"  Max: {np.max(real_probs):.4f}")

    print(f"\nFake Images (should be > 0.5):")
    print(f"  Mean: {np.mean(fake_probs):.4f}")
    print(f"  Median: {np.median(fake_probs):.4f}")
    print(f"  Min: {np.min(fake_probs):.4f}")
    print(f"  Max: {np.max(fake_probs):.4f}")

if __name__ == '__main__':
    print("="*60)
    print("Visualizing Misclassifications")
    print("="*60)

    print("\n1. Finding and visualizing misclassified images...")
    try:
        visualize_misclassifications()
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. Analyzing confidence distribution...")
    try:
        analyze_confidence_distribution()
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)