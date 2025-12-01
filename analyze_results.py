import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch.utils.data import DataLoader
import open_clip
from PIL import Image

def plot_training_curves(csv_file='results.csv'):
    """Plot training curves"""
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(df['epoch'], df['train_loss'], marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    ax2.plot(df['epoch'], df['test_acc'], marker='o', label='Accuracy')
    ax2.plot(df['epoch'], df['test_auc'], marker='s', label='AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Test Performance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    print("✓ Saved training_curves.png")

def visualize_feature_space(model_path='best_cifake_clip_classifier.pt'):
    """Visualize CLIP feature space with t-SNE"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    model = model.to(DEVICE)
    model.eval()

    # Load test data
    from train_cifake_clip import CIFAKEDataset
    test_dataset = CIFAKEDataset('cifake_test.csv', transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Extract features
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            feat = model.encode_image(images.to(DEVICE))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.vstack(features)
    labels = np.array(labels)

    # Sample for faster t-SNE
    n_samples = min(5000, len(features))
    idx = np.random.choice(len(features), n_samples, replace=False)
    features_sample = features[idx]
    labels_sample = labels[idx]

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sample)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[labels_sample==0, 0],
                features_2d[labels_sample==0, 1],
                c='blue', label='Real', alpha=0.5, s=10)
    plt.scatter(features_2d[labels_sample==1, 0],
                features_2d[labels_sample==1, 1],
                c='red', label='Fake (AI-generated)', alpha=0.5, s=10)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('CLIP Feature Space: Real vs AI-Generated Images')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_space_tsne.png', dpi=300)
    print("✓ Saved feature_space_tsne.png")

def compare_models():
    """Compare all models"""
    results = {
        'Model': ['Zero-Shot CLIP', 'Fine-tuned CLIP', 'ResNet50'],
        'AUC': [0.0, 0.0, 0.0],  # Fill from your results
        'Accuracy': [0.0, 0.0, 0.0]
    }

    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(df['Model'], df['AUC'])
    ax1.set_ylabel('AUC')
    ax1.set_title('Model Comparison - AUC')
    ax1.set_ylim([0, 1])

    ax2.bar(df['Model'], df['Accuracy'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Comparison - Accuracy')
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("✓ Saved model_comparison.png")

if __name__ == '__main__':
    print("Generating analysis plots...")
    plot_training_curves()
    visualize_feature_space()
    compare_models()
    print("\nDone!")