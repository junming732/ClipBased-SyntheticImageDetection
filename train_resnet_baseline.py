#!/usr/bin/env python3
"""
ResNet50 Baseline for AFHQ Real vs Fake Detection
Standard CNN baseline for comparison
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm
import pandas as pd

class ResNetClassifier(nn.Module):
    """ResNet50 binary classifier"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, auc

def main():
    # Configuration
    BATCH_SIZE = 64  # Smaller batch for ResNet50
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    print(f"Using device: {DEVICE}")

    # ResNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\nLoading datasets...")
    from train_afhq_clip import AFHQDataset
    train_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='train', transform=transform)
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print("\nCreating ResNet50 model...")
    model = ResNetClassifier(pretrained=True).to(DEVICE)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "="*60)
    print("Starting ResNet50 training...")
    print("="*60)

    best_val_acc = 0.0
    results = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # Save results
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': val_acc,
            'test_auc': val_auc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, 'best_resnet_afhq.pth')
            print(f"✓ Saved new best model (acc={val_acc:.4f})")

    # Save training results
    results_df = pd.DataFrame(results)
    results_df.to_csv('resnet_training_results.csv', index=False)
    print("\n✓ Saved training results to resnet_training_results.csv")

    print("\n" + "="*60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()