#!/usr/bin/env python3
"""
CORRECTED + ROBUST: Fine-tune CLIP for AFHQ with proper train/val split and corrupted image handling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import warnings

class AFHQDataset(Dataset):
    """
    CORRECTED + ROBUST: Loads AFHQ with proper train/val split and skips corrupted images
    """
    def __init__(self, real_root, fake_root, split='train', transform=None):
        self.samples = []
        self.transform = transform

        # Real images from AFHQv2
        real_path = Path(real_root) / split
        print(f"Loading real images from: {real_path}")
        real_count = 0
        for animal_type in ['cat', 'dog', 'wild']:
            animal_dir = real_path / animal_type
            if animal_dir.exists():
                for img_file in animal_dir.glob('*.jpg'):
                    # Verify image is valid
                    if self._verify_image(img_file):
                        self.samples.append((str(img_file), 0))
                        real_count += 1

        print(f"Loaded {real_count} real images")

        # Fake images from StyleGAN3
        fake_path = Path(fake_root)
        print(f"Loading fake images from: {fake_path}")
        fake_count = 0
        corrupted_count = 0

        for stylegan_dir in fake_path.iterdir():
            if stylegan_dir.is_dir() and 'afhqv2' in stylegan_dir.name:
                print(f"  Loading from: {stylegan_dir.name}")
                for img_file in stylegan_dir.glob('*.png'):
                    # Verify image is valid
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

        # Robust image loading with retry
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image fails to load, return a black image (won't affect training much)
            print(f"⚠️  Failed to load {img_path}: {e}")
            image = Image.new('RGB', (512, 512), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label

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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc

def main():
    # CORRECTED PATHS
    REAL_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/stargan-v2/data/afhq'
    TRAIN_FAKE_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'
    VAL_FAKE_ROOT = '/crex/proj/uppmax2025-2-346/nobackup/private/junming/FakeImageDataset/ImageData/val/stylegan3-60K/stylegan3-60K'

    # Training configuration
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_WORKERS = 2  # Reduced from 4 to avoid worker issues

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Load CLIP model
    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    model = model.to(DEVICE)

    # Create classifier
    feature_dim = 512  # ViT-B-32 feature dimension
    classifier = CLIPClassifier(model, feature_dim).to(DEVICE)

    # CORRECTED: Different fake roots for train and val!
    print("\nCreating datasets...")
    train_dataset = AFHQDataset(REAL_ROOT, TRAIN_FAKE_ROOT, split='train', transform=preprocess)
    val_dataset = AFHQDataset(REAL_ROOT, VAL_FAKE_ROOT, split='val', transform=preprocess)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
            }, 'best_afhq_clip_classifier_CORRECTED.pth')
            print(f"✓ Saved new best model (acc={val_acc:.4f})")

    print("\n" + "="*60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)
    print("Training completed!")

if __name__ == '__main__':
    main()