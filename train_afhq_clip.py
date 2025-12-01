#!/usr/bin/env python3
"""
Fine-tune CLIP for AFHQ Real vs Fake Detection
Trains a linear classifier on frozen CLIP features
Real images: AFHQv2 dataset
Fake images: StyleGAN3-generated images
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import os
from pathlib import Path

class AFHQDataset(Dataset):
    """Dataset for AFHQ real + StyleGAN3 fake images"""
    def __init__(self, real_root, fake_root, split='train', transform=None):
        self.transform = transform
        self.samples = []

        real_path = Path(real_root) / split
        # AFHQv2 has cat, dog, wild subfolders
        for animal_type in ['cat', 'dog', 'wild']:
            animal_dir = real_path / animal_type
            if animal_dir.exists():
                for img_file in animal_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), 0))  # 0 = real

        # StyleGAN3 fake images - only AFHQ-related ones
        fake_path = Path(fake_root)
        # Look for images in AFHQ subdirectories only
        for stylegan_dir in fake_path.iterdir():
            if stylegan_dir.is_dir() and 'afhqv2' in stylegan_dir.name:
                for img_file in stylegan_dir.glob('*.png'):
                    self.samples.append((str(img_file), 1))  # 1 = fake
                # Also check for jpg files
                for img_file in stylegan_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), 1))  # 1 = fake

        print(f"Loaded {len([s for s in self.samples if s[1]==0])} real images")
        print(f"Loaded {len([s for s in self.samples if s[1]==1])} fake images")
        print(f"Total: {len(self.samples)} images for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class CLIPClassifier(nn.Module):
    """Linear classifier on frozen CLIP features"""
    def __init__(self, clip_model, feature_dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(feature_dim, 1)

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classifier(features.float())
        return logits

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

        # For metrics
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

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return avg_loss, acc, auc

def main():
    # Paths - adjust these to your setup
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    # Hyperparameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Load CLIP
    print("Loading CLIP model...")
    model_name = 'ViT-B-32'
    pretrained = 'openai'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(DEVICE)

    # Create classifier
    feature_dim = 512  # ViT-B-32 feature dimension
    classifier = CLIPClassifier(clip_model, feature_dim).to(DEVICE)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='train', transform=preprocess)
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=preprocess)

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

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(
            classifier, train_loader, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_acc, val_auc = evaluate(
            classifier, val_loader, criterion, DEVICE
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_auc': val_auc,
            }, 'best_afhq_clip_classifier.pth')
            print(f"✓ Saved new best model (acc={val_acc:.4f})")

    print("\n" + "="*60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()