"""
Threshold Optimization Script for Precision-Recall Trade-off

This script helps you find the optimal threshold to balance precision and recall
based on your specific requirements.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)
from predict_anomalies import AnomalyPredictor
import warnings
warnings.filterwarnings('ignore')


def find_optimal_threshold(predictor, data_path, turbine_id, 
                          target_precision=0.8, target_recall=0.9):
    """
    Find optimal threshold based on target precision or recall
    
    Args:
        predictor: AnomalyPredictor instance
        data_path: Path to data
        turbine_id: Turbine ID to analyze
        target_precision: Desired precision (default: 0.8)
        target_recall: Desired recall (default: 0.9)
    
    Returns:
        results: Dictionary with threshold analysis
    """
    print("="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Get predictions with probabilities
    predictions, probabilities, original_data, sequence_indices = predictor.predict_from_file(
        data_path=data_path,
        turbine_id=turbine_id
    )
    
    # Load true labels
    event_info = pd.read_csv('dataset_dl/Wind Farm A/event_info.csv', sep=';')
    turbine_events = event_info[event_info['asset'] == int(turbine_id)]
    
    true_labels = np.zeros(len(predictions))
    for _, event in turbine_events.iterrows():
        if event['event_label'] == 'anomaly':
            start_idx = event['event_start_id']
            end_idx = event['event_end_id']
            for i, (seq_start, seq_end) in enumerate(sequence_indices):
                if not (seq_end < start_idx or seq_start > end_idx):
                    true_labels[i] = 1
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
    
    # Find threshold for target precision
    precision_mask = precision >= target_precision
    if precision_mask.any():
        idx_precision = np.where(precision_mask)[0][0]
        threshold_for_precision = thresholds[idx_precision] if idx_precision < len(thresholds) else thresholds[-1]
        recall_at_target_precision = recall[idx_precision]
    else:
        threshold_for_precision = None
        recall_at_target_precision = None
    
    # Find threshold for target recall
    recall_mask = recall >= target_recall
    if recall_mask.any():
        idx_recall = np.where(recall_mask)[0][-1]
        threshold_for_recall = thresholds[idx_recall] if idx_recall < len(thresholds) else thresholds[-1]
        precision_at_target_recall = precision[idx_recall]
    else:
        threshold_for_recall = None
        precision_at_target_recall = None
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    best_f1_score = f1_scores[best_f1_idx]
    
    # Find threshold that balances precision and recall (closest to equal)
    balance_diff = np.abs(precision - recall)
    balance_idx = np.argmin(balance_diff)
    balanced_threshold = thresholds[balance_idx] if balance_idx < len(thresholds) else thresholds[-1]
    
    results = {
        'probabilities': probabilities,
        'true_labels': true_labels,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'current_threshold': predictor.threshold,
        'best_f1_threshold': best_f1_threshold,
        'best_f1_score': best_f1_score,
        'balanced_threshold': balanced_threshold,
        'threshold_for_precision': threshold_for_precision,
        'recall_at_target_precision': recall_at_target_precision,
        'threshold_for_recall': threshold_for_recall,
        'precision_at_target_recall': precision_at_target_recall,
        'target_precision': target_precision,
        'target_recall': target_recall
    }
    
    return results


def print_threshold_analysis(results):
    """Print detailed threshold analysis"""
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📊 Current Threshold: {results['current_threshold']:.4f}")
    
    print(f"\n🎯 Best F1 Score Threshold: {results['best_f1_threshold']:.4f}")
    print(f"   F1 Score: {results['best_f1_score']:.4f}")
    
    print(f"\n⚖️  Balanced Precision-Recall Threshold: {results['balanced_threshold']:.4f}")
    
    if results['threshold_for_precision'] is not None:
        print(f"\n🎯 For Target Precision ≥ {results['target_precision']:.2f}:")
        print(f"   Threshold: {results['threshold_for_precision']:.4f}")
        print(f"   Recall at this threshold: {results['recall_at_target_precision']:.4f}")
    else:
        print(f"\n⚠️  Cannot achieve target precision of {results['target_precision']:.2f}")
    
    if results['threshold_for_recall'] is not None:
        print(f"\n🎯 For Target Recall ≥ {results['target_recall']:.2f}:")
        print(f"   Threshold: {results['threshold_for_recall']:.4f}")
        print(f"   Precision at this threshold: {results['precision_at_target_recall']:.4f}")
    else:
        print(f"\n⚠️  Cannot achieve target recall of {results['target_recall']:.2f}")
    
    print("\n" + "="*60)


def evaluate_with_threshold(probabilities, true_labels, threshold):
    """Evaluate predictions with a specific threshold"""
    predictions = (probabilities >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0)
    }
    
    return metrics, predictions


def plot_threshold_analysis(results, save_path='results/threshold_analysis.png'):
    """Create comprehensive threshold analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision-Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(results['recall'], results['precision'], 'b-', linewidth=2)
    ax1.axhline(y=results['target_precision'], color='r', linestyle='--', 
                label=f'Target Precision ({results["target_precision"]:.2f})')
    ax1.axvline(x=results['target_recall'], color='g', linestyle='--',
                label=f'Target Recall ({results["target_recall"]:.2f})')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision, Recall, F1 vs Threshold
    ax2 = axes[0, 1]
    # Extend arrays to match thresholds length
    precision_plot = results['precision'][:-1]
    recall_plot = results['recall'][:-1]
    f1_plot = results['f1_scores'][:-1]
    
    ax2.plot(results['thresholds'], precision_plot, 'b-', label='Precision', linewidth=2)
    ax2.plot(results['thresholds'], recall_plot, 'g-', label='Recall', linewidth=2)
    ax2.plot(results['thresholds'], f1_plot, 'r-', label='F1 Score', linewidth=2)
    ax2.axvline(x=results['current_threshold'], color='orange', linestyle='--',
                label=f'Current ({results["current_threshold"]:.3f})', linewidth=2)
    ax2.axvline(x=results['best_f1_threshold'], color='purple', linestyle=':',
                label=f'Best F1 ({results["best_f1_threshold"]:.3f})', linewidth=2)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: Threshold Recommendations
    ax3 = axes[1, 0]
    thresholds_to_compare = [
        ('Current', results['current_threshold']),
        ('Best F1', results['best_f1_threshold']),
        ('Balanced', results['balanced_threshold'])
    ]
    
    if results['threshold_for_precision'] is not None:
        thresholds_to_compare.append(
            (f'Precision≥{results["target_precision"]:.2f}', results['threshold_for_precision'])
        )
    
    metrics_comparison = []
    labels = []
    for name, thresh in thresholds_to_compare:
        metrics, _ = evaluate_with_threshold(results['probabilities'], 
                                            results['true_labels'], thresh)
        metrics_comparison.append([metrics['precision'], metrics['recall'], metrics['f1']])
        labels.append(f'{name}\n({thresh:.3f})')
    
    metrics_comparison = np.array(metrics_comparison)
    x = np.arange(len(labels))
    width = 0.25
    
    ax3.bar(x - width, metrics_comparison[:, 0], width, label='Precision', color='blue', alpha=0.7)
    ax3.bar(x, metrics_comparison[:, 1], width, label='Recall', color='green', alpha=0.7)
    ax3.bar(x + width, metrics_comparison[:, 2], width, label='F1', color='red', alpha=0.7)
    
    ax3.set_xlabel('Threshold Strategy', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Threshold Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1])
    
    # Plot 4: Probability Distribution
    ax4 = axes[1, 1]
    normal_probs = results['probabilities'][results['true_labels'] == 0]
    anomaly_probs = results['probabilities'][results['true_labels'] == 1]
    
    ax4.hist(normal_probs, bins=50, alpha=0.6, label='Normal', color='green', edgecolor='black')
    ax4.hist(anomaly_probs, bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
    ax4.axvline(x=results['current_threshold'], color='orange', linestyle='--',
                label=f'Current Threshold ({results["current_threshold"]:.3f})', linewidth=2)
    ax4.axvline(x=results['best_f1_threshold'], color='purple', linestyle=':',
                label=f'Best F1 Threshold ({results["best_f1_threshold"]:.3f})', linewidth=2)
    ax4.set_xlabel('Anomaly Probability', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Threshold analysis plot saved to {save_path}")
    plt.close()


def main():
    """Main function to run threshold optimization"""
    
    # Initialize predictor
    predictor = AnomalyPredictor('results/hierarchical_transformer_lstm_v2.pth')
    
    # Configuration
    data_path = 'dataset_dl/Wind Farm A/datasets'
    turbine_id = '0'
    target_precision = 0.80  # Adjust this to your desired precision
    target_recall = 0.90     # Adjust this to your desired recall
    
    print(f"\n🎯 Target Precision: {target_precision:.2f}")
    print(f"🎯 Target Recall: {target_recall:.2f}")
    
    # Find optimal threshold
    results = find_optimal_threshold(
        predictor, 
        data_path, 
        turbine_id,
        target_precision=target_precision,
        target_recall=target_recall
    )
    
    # Print analysis
    print_threshold_analysis(results)
    
    # Plot analysis
    plot_threshold_analysis(results)
    
    # Detailed comparison of different thresholds
    print("\n" + "="*60)
    print("DETAILED THRESHOLD COMPARISON")
    print("="*60)
    
    thresholds_to_test = {
        'Current': results['current_threshold'],
        'Best F1': results['best_f1_threshold'],
        'Balanced': results['balanced_threshold'],
    }
    
    if results['threshold_for_precision'] is not None:
        thresholds_to_test[f'Precision≥{target_precision:.2f}'] = results['threshold_for_precision']
    
    # Add some manual thresholds to test
    thresholds_to_test.update({
        'Conservative (0.3)': 0.3,
        'Moderate (0.5)': 0.5,
        'Aggressive (0.7)': 0.7
    })
    
    print(f"\n{'Strategy':<25} {'Threshold':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 77)
    
    for name, threshold in thresholds_to_test.items():
        metrics, _ = evaluate_with_threshold(
            results['probabilities'],
            results['true_labels'],
            threshold
        )
        print(f"{name:<25} {threshold:<12.4f} {metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if results['threshold_for_precision'] is not None:
        print(f"\n✅ To achieve precision ≥ {target_precision:.2f}:")
        print(f"   Use threshold: {results['threshold_for_precision']:.4f}")
        print(f"   Expected recall: {results['recall_at_target_precision']:.4f}")
    
    print(f"\n✅ For balanced precision-recall:")
    print(f"   Use threshold: {results['balanced_threshold']:.4f}")
    
    print(f"\n✅ For maximum F1 score:")
    print(f"   Use threshold: {results['best_f1_threshold']:.4f}")
    
    print("\n💡 To use a new threshold in predictions, update the threshold in")
    print("   predict_anomalies.py or pass it when creating predictions.")
    print("="*60)


if __name__ == "__main__":
    main()
