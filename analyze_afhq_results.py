#!/usr/bin/env python3
"""
Analyze AFHQ CLIP Training Results
Generates visualizations for training curves, feature space, and model comparisons
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch.utils.data import DataLoader
import open_clip
from PIL import Image
from pathlib import Path

def plot_training_curves(csv_file='training_results.csv'):
    """Plot training curves"""
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df['epoch'], df['train_loss'], marker='o', color='steelblue', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss on AFHQ Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['epoch'], df['test_acc'], marker='o', label='Accuracy', linewidth=2)
    ax2.plot(df['epoch'], df['test_auc'], marker='s', label='AUC', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('afhq_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved afhq_training_curves.png")
    plt.close()

def visualize_feature_space(model_path='best_afhq_clip_classifier.pth'):
    """Visualize CLIP feature space with t-SNE for AFHQ dataset"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    print(f"Using device: {DEVICE}")

    # Load CLIP model (use same as training)
    print("Loading CLIP model...")
    model_name = 'ViT-B-32'
    pretrained = 'openai'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    # Load validation dataset
    from train_afhq_clip import AFHQDataset
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=preprocess)
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )

    print(f"Loaded {len(val_dataset)} validation images")

    # Extract features
    print("Extracting CLIP features...")
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in val_loader:
            feat = clip_model.encode_image(images.to(DEVICE))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    print(f"Extracted {len(features)} features")

    # Sample for faster t-SNE (use all if dataset is small enough)
    n_samples = min(3000, len(features))
    idx = np.random.choice(len(features), n_samples, replace=False)
    features_sample = features[idx]
    labels_sample = labels[idx]

    # t-SNE
    print(f"Running t-SNE on {n_samples} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features_sample)

    # Plot
    plt.figure(figsize=(10, 8))

    # Real images (blue)
    real_mask = labels_sample == 0
    plt.scatter(features_2d[real_mask, 0],
                features_2d[real_mask, 1],
                c='#2E86AB', label='Real (AFHQv2)', alpha=0.6, s=20, edgecolors='none')

    # Fake images (red)
    fake_mask = labels_sample == 1
    plt.scatter(features_2d[fake_mask, 0],
                features_2d[fake_mask, 1],
                c='#A23B72', label='Fake (StyleGAN3)', alpha=0.6, s=20, edgecolors='none')

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('CLIP Feature Space: Real vs StyleGAN3-Generated Animal Faces',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('afhq_feature_space_tsne.png', dpi=300, bbox_inches='tight')
    print("✓ Saved afhq_feature_space_tsne.png")
    plt.close()

def compare_models(results_file='model_comparison_results.csv'):
    """Compare different models on AFHQ dataset"""
    # You'll need to fill these in with actual results
    # This is a template - update with your real numbers
    results = {
        'Model': [
            'Zero-Shot CLIP\n(text prompts)',
            'Pre-trained Detector\n(high-res trained)',
            'Fine-tuned CLIP\n(AFHQ)'
        ],
        'AUC': [0.50, 0.52, 0.95],  # Update with actual results
        'Accuracy': [0.50, 0.51, 0.93]  # Update with actual results
    }

    df = pd.DataFrame(results)

    # Save results
    df.to_csv(results_file, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#E63946', '#F77F00', '#06A77D']

    # AUC comparison
    bars1 = ax1.bar(df['Model'], df['AUC'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('Model Comparison - AUC Score', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Accuracy comparison
    bars2 = ax2.bar(df['Model'], df['Accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('afhq_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved afhq_model_comparison.png")
    print(f"✓ Saved results to {results_file}")
    plt.close()

def analyze_per_class_performance(model_path='best_afhq_clip_classifier.pth'):
    """Analyze performance on different animal types (cat/dog/wild)"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'

    print("Loading CLIP model...")
    model_name = 'ViT-B-32'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai'
    )
    clip_model = clip_model.to(DEVICE)

    # Load classifier
    from train_afhq_clip import CLIPClassifier
    classifier = CLIPClassifier(clip_model, feature_dim=512).to(DEVICE)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=DEVICE)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: {model_path} not found, using untrained classifier")

    classifier.eval()

    # Analyze each animal type
    animal_types = ['cat', 'dog', 'wild']
    results = {'Animal Type': [], 'Accuracy': [], 'Count': []}

    for animal_type in animal_types:
        animal_dir = Path(REAL_ROOT) / 'val' / animal_type
        if not animal_dir.exists():
            continue

        images = list(animal_dir.glob('*.jpg'))
        correct = 0

        with torch.no_grad():
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

                output = classifier(img_tensor).squeeze()
                pred = torch.sigmoid(output) > 0.5  # Should predict 0 for real

                if pred == 0:  # Correctly identified as real
                    correct += 1

        accuracy = correct / len(images) if len(images) > 0 else 0
        results['Animal Type'].append(animal_type.capitalize())
        results['Accuracy'].append(accuracy)
        results['Count'].append(len(images))
        print(f"{animal_type.capitalize()}: {accuracy:.2%} ({correct}/{len(images)})")

    # Plot
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Animal Type'], df['Accuracy'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Accuracy (Real Detection)', fontsize=12)
    ax.set_xlabel('Animal Type', fontsize=12)
    ax.set_title('Real Image Detection Accuracy by Animal Type',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, count in zip(bars, df['Count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}\n(n={count})',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('afhq_per_class_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved afhq_per_class_performance.png")
    plt.close()

if __name__ == '__main__':
    print("="*60)
    print("Generating AFHQ Analysis Plots")
    print("="*60)

    print("\n1. Plotting training curves...")
    try:
        plot_training_curves()
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. Visualizing feature space...")
    try:
        visualize_feature_space()
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Comparing models...")
    try:
        compare_models()
    except Exception as e:
        print(f"   Error: {e}")

    print("\n4. Analyzing per-class performance...")
    try:
        analyze_per_class_performance()
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)