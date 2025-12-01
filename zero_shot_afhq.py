#!/usr/bin/env python3
"""
Zero-Shot CLIP Evaluation on AFHQ
Uses text prompts without any training
"""
import torch
import open_clip
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm

def zero_shot_evaluation():
    """Evaluate zero-shot CLIP with text prompts"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    REAL_ROOT = '/home/junming/nobackup_junming/stargan-v2/data/afhq'
    FAKE_ROOT = '/home/junming/nobackup_junming/FakeImageDataset/ImageData/train/stylegan3-80K/stylegan3-80K'

    print(f"Using device: {DEVICE}")

    # Load CLIP
    print("Loading CLIP model...")
    model_name = 'ViT-B-32'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai'
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()

    # Create text prompts
    text_prompts = [
        "a real photograph",
        "an AI-generated synthetic image"
    ]

    print(f"\nText prompts:")
    print(f"  [0] {text_prompts[0]}")
    print(f"  [1] {text_prompts[1]}")

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(text_prompts).to(DEVICE)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load validation dataset
    from train_afhq_clip import AFHQDataset
    val_dataset = AFHQDataset(REAL_ROOT, FAKE_ROOT, split='val', transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    print(f"\nEvaluating on {len(val_dataset)} validation images...")

    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Zero-shot evaluation"):
            images = images.to(DEVICE)

            # Get image features
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with text prompts
            similarity = (image_features @ text_features.T) * 100  # Scale for softmax
            probs = similarity.softmax(dim=-1)

            # Get probability of "AI-generated" (index 1)
            fake_probs = probs[:, 1].cpu().numpy()
            preds = (fake_probs > 0.5).astype(int)

            all_probs.extend(fake_probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n" + "="*60)
    print("ZERO-SHOT CLIP RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

    # Save results
    results = {
        'model': 'Zero-Shot CLIP',
        'accuracy': accuracy,
        'auc': auc
    }

    import json
    with open('zero_shot_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Saved results to zero_shot_results.json")

    return accuracy, auc

if __name__ == '__main__':
    zero_shot_evaluation()