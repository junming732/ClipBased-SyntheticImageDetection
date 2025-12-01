#!/usr/bin/env python3
"""
Analyze zero-shot CLIP detection results on CIFAKE
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve
)

def load_and_prepare_data(csv_file):
    """Load results and prepare labels"""
    df = pd.read_csv(csv_file)

    # Extract true labels from filename
    # Real images have 'REAL' in path, fake have 'FAKE'
    df['true_label'] = df['filename'].apply(
        lambda x: 0 if 'REAL' in x else 1  # 0=real, 1=fake
    )

    # The score columns contain log-likelihood ratios (LLR)
    # Positive LLR = classified as synthetic/fake
    # We'll use the 'fusion' column as the main score
    score_columns = [col for col in df.columns if col not in ['filename', 'true_label']]

    print(f"Loaded {len(df)} images")
    print(f"Score columns: {score_columns}")
    print(f"Real images: {sum(df['true_label']==0)}")
    print(f"Fake images: {sum(df['true_label']==1)}")

    return df, score_columns

def plot_score_distributions(df, score_col='fusion', output='score_distribution.png'):
    """Plot distribution of scores for real vs fake"""
    real_scores = df[df['true_label']==0][score_col]
    fake_scores = df[df['true_label']==1][score_col]

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real', color='blue', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake (AI-generated)', color='red', density=True)
    plt.axvline(x=0, color='black', linestyle='--', label='Decision boundary (LLR=0)')
    plt.xlabel('Log-Likelihood Ratio (LLR)')
    plt.ylabel('Density')
    plt.title(f'Score Distribution: Real vs Fake Images\n({score_col} detector)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"✓ Saved {output}")

def plot_roc_curve(df, score_columns, output='roc_curve.png'):
    """Plot ROC curves for all detectors"""
    plt.figure(figsize=(10, 8))

    results = {}
    for score_col in score_columns:
        y_true = df['true_label']
        y_score = df[score_col]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        results[score_col] = roc_auc

        plt.plot(fpr, tpr, lw=2, label=f'{score_col} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: CLIP-based Detection on CIFAKE')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"✓ Saved {output}")

    return results

def plot_precision_recall(df, score_col='fusion', output='precision_recall.png'):
    """Plot precision-recall curve"""
    y_true = df['true_label']
    y_score = df[score_col]

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"✓ Saved {output}")

def plot_confusion_matrix(df, score_col='fusion', output='confusion_matrix.png'):
    """Plot confusion matrix"""
    y_true = df['true_label']
    y_pred = (df[score_col] > 0).astype(int)  # Threshold at 0

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix\n({score_col} detector, threshold=0)')

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"✓ Saved {output}")

def print_metrics(df, score_columns):
    """Print detailed metrics for all detectors"""
    print("\n" + "="*70)
    print("ZERO-SHOT CLIP DETECTION RESULTS ON CIFAKE")
    print("="*70)

    for score_col in score_columns:
        y_true = df['true_label']
        y_score = df[score_col]
        y_pred = (y_score > 0).astype(int)

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Accuracy
        accuracy = (y_pred == y_true).mean()

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{score_col.upper()} Detector:")
        print(f"  ROC AUC:    {roc_auc:.4f}")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1-Score:   {f1:.4f}")
        print(f"  True Pos:   {tp:,} / {tp+fn:,}")
        print(f"  True Neg:   {tn:,} / {tn+fp:,}")
        print(f"  False Pos:  {fp:,} (Real→Fake)")
        print(f"  False Neg:  {fn:,} (Fake→Real)")

    print("\n" + "="*70)

def compare_detectors(df, score_columns, output='detector_comparison.png'):
    """Compare different detectors"""
    metrics = []

    for score_col in score_columns:
        y_true = df['true_label']
        y_score = df[score_col]
        y_pred = (y_score > 0).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        accuracy = (y_pred == y_true).mean()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics.append({
            'Detector': score_col,
            'AUC': roc_auc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })

    metrics_df = pd.DataFrame(metrics)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, metric in enumerate(['AUC', 'Accuracy', 'Precision', 'Recall']):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(metrics_df['Detector'], metrics_df[metric],
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(metrics_df)])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"✓ Saved {output}")

    return metrics_df

def analyze_misclassifications(df, score_col='fusion', n_samples=10):
    """Show examples of misclassifications"""
    y_true = df['true_label']
    y_pred = (df[score_col] > 0).astype(int)

    # False positives (real classified as fake)
    fp_mask = (y_true == 0) & (y_pred == 1)
    false_positives = df[fp_mask].nlargest(n_samples, score_col)

    # False negatives (fake classified as real)
    fn_mask = (y_true == 1) & (y_pred == 0)
    false_negatives = df[fn_mask].nsmallest(n_samples, score_col)

    print(f"\n{'='*70}")
    print(f"MISCLASSIFICATION ANALYSIS ({score_col})")
    print(f"{'='*70}")
    print(f"\nFalse Positives (Real→Fake): {fp_mask.sum():,}")
    print(f"Top {min(n_samples, len(false_positives))} examples:")
    for idx, row in false_positives.iterrows():
        print(f"  Score: {row[score_col]:7.2f} | {row['filename'].split('/')[-1]}")

    print(f"\nFalse Negatives (Fake→Real): {fn_mask.sum():,}")
    print(f"Top {min(n_samples, len(false_negatives))} examples:")
    for idx, row in false_negatives.iterrows():
        print(f"  Score: {row[score_col]:7.2f} | {row['filename'].split('/')[-1]}")

def main():
    import sys

    # Get CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'cifake_zero_shot_results.csv'

    print(f"Analyzing: {csv_file}\n")

    # Load data
    df, score_columns = load_and_prepare_data(csv_file)

    # Use fusion as primary detector
    primary_detector = 'fusion' if 'fusion' in score_columns else score_columns[0]

    # Generate all plots
    print("\nGenerating visualizations...")
    plot_score_distributions(df, primary_detector)
    plot_roc_curve(df, score_columns)
    plot_precision_recall(df, primary_detector)
    plot_confusion_matrix(df, primary_detector)

    if len(score_columns) > 1:
        metrics_df = compare_detectors(df, score_columns)
        print("\nDetector Comparison:")
        print(metrics_df.to_string(index=False))

    # Print metrics
    print_metrics(df, score_columns)

    # Analyze errors
    analyze_misclassifications(df, primary_detector)

    print("\n" + "="*70)
    print("✓ Analysis complete! Check the generated PNG files.")
    print("="*70)

if __name__ == '__main__':
    main()