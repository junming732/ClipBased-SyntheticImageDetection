import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

class CIFAKEDataset(Dataset):
    """Dataset for CNN baseline"""
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

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load ResNet50
    print("Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    # Datasets
    train_dataset = CIFAKEDataset('cifake_train.csv', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    test_dataset = CIFAKEDataset('cifake_test.csv', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    for epoch in range(5):
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        auc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_resnet50_baseline.pt')

    print(f"\nBest ResNet50 AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()