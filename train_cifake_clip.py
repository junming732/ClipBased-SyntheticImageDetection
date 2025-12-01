import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import open_clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import os

class CIFAKEDataset(Dataset):
    """Dataset for CIFAKE with CLIP preprocessing"""
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filename']
        label = 1 if self.df.iloc[idx]['typ'] == 'fake' else 0

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

class CLIPClassifier(nn.Module):
    """Linear classifier on frozen CLIP features"""
    def __init__(self, clip_model, feature_dim=768):
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
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, auc, all_labels, all_preds

def main():
    # Configuration
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Load pretrained CLIP
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    model = model.to(DEVICE)

    # Freeze CLIP encoder
    for param in model.parameters():
        param.requires_grad = False

    # Create classifier
    classifier = CLIPClassifier(model).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in classifier.classifier.parameters()):,}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CIFAKEDataset('cifake_train.csv', transform=preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataset = CIFAKEDataset('cifake_test.csv', transform=preprocess)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training setup
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_auc = 0
    results = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss = train_epoch(classifier, train_loader, criterion, optimizer, DEVICE)
        acc, auc, labels, preds = evaluate(classifier, test_loader, DEVICE)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test AUC: {auc:.4f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(classifier.classifier.state_dict(), 'best_cifake_clip_classifier.pt')
            print(f"✓ Saved best model (AUC: {auc:.4f})")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_acc': acc,
            'test_auc': auc
        })

    # Final evaluation on best model
    print(f"\n{'='*60}")
    print("Final Evaluation (Best Model)")
    print(f"{'='*60}")

    classifier.classifier.load_state_dict(torch.load('best_cifake_clip_classifier.pt'))
    acc, auc, labels, preds = evaluate(classifier, test_loader, DEVICE)

    print(f"\nBest Test Accuracy: {acc:.4f}")
    print(f"Best Test AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('training_results.csv', index=False)
    print("\n✓ Saved training results to training_results.csv")

if __name__ == '__main__':
    main()