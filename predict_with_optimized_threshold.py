"""
Prediction Script with Optimized Threshold for Better Precision

This script uses the optimized threshold (0.9662) for better precision-recall balance.
"""

from predict_anomalies import AnomalyPredictor
import numpy as np
import pandas as pd


def predict_with_custom_threshold(data_path, turbine_id, threshold=0.9662):
    """
    Make predictions with custom threshold
    
    Args:
        data_path: Path to data directory
        turbine_id: Turbine ID to analyze
        threshold: Custom threshold (default: 0.9662 for best F1)
    """
    # Initialize predictor
    predictor = AnomalyPredictor('results/hierarchical_transformer_lstm_v2.pth')
    
    # Override the threshold
    original_threshold = predictor.threshold
    predictor.threshold = threshold
    
    print(f"Original threshold: {original_threshold:.4f}")
    print(f"Using optimized threshold: {threshold:.4f}")
    
    # Make predictions
    predictions, probabilities, original_data, sequence_indices = predictor.predict_from_file(
        data_path=data_path,
        turbine_id=turbine_id
    )
    
    # Visualize with new threshold
    predictor.visualize_predictions(
        predictions, 
        probabilities, 
        sequence_indices,
        original_data,
        save_path=f'results/predictions_turbine_{turbine_id}_optimized.png'
    )
    
    # Evaluate if true labels available
    try:
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
        
        print("\n" + "="*60)
        print("COMPARISON: Original vs Optimized Threshold")
        print("="*60)
        
        # Evaluate with original threshold
        preds_original = (probabilities >= original_threshold).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        print(f"\n📊 Original Threshold ({original_threshold:.4f}):")
        print(f"   Accuracy:  {accuracy_score(true_labels, preds_original):.4f}")
        print(f"   Precision: {precision_score(true_labels, preds_original, zero_division=0):.4f}")
        print(f"   Recall:    {recall_score(true_labels, preds_original, zero_division=0):.4f}")
        print(f"   F1 Score:  {f1_score(true_labels, preds_original, zero_division=0):.4f}")
        print(f"   Anomaly Rate: {preds_original.sum() / len(preds_original) * 100:.2f}%")
        
        # Evaluate with optimized threshold
        print(f"\n📊 Optimized Threshold ({threshold:.4f}):")
        metrics = predictor.evaluate_predictions(predictions, true_labels)
        print(f"   Anomaly Rate: {predictions.sum() / len(predictions) * 100:.2f}%")
        
        print("\n✅ Improvement:")
        precision_improvement = (precision_score(true_labels, predictions, zero_division=0) - 
                                precision_score(true_labels, preds_original, zero_division=0)) * 100
        recall_change = (recall_score(true_labels, predictions, zero_division=0) - 
                        recall_score(true_labels, preds_original, zero_division=0)) * 100
        
        print(f"   Precision: {precision_improvement:+.2f}%")
        print(f"   Recall: {recall_change:+.2f}%")
        
    except Exception as e:
        print(f"\nCould not evaluate with true labels: {e}")
    
    return predictions, probabilities


def main():
    """Main function"""
    print("="*60)
    print("PREDICTIONS WITH OPTIMIZED THRESHOLD")
    print("="*60)
    
    # Configuration
    data_path = 'dataset_dl/Wind Farm A/datasets'
    turbine_id = '0'
    
    # Choose your threshold strategy:
    # - 0.9662: Best F1 (recommended) - good balance
    # - 0.9768: Balanced precision-recall
    # - 0.9805: High precision (≥80%)
    
    threshold = 0.9662  # Best F1 threshold
    
    print(f"\n🎯 Analyzing Turbine {turbine_id}")
    print(f"🎯 Using threshold: {threshold:.4f} (Best F1 Strategy)")
    
    predictions, probabilities = predict_with_custom_threshold(
        data_path, 
        turbine_id, 
        threshold=threshold
    )
    
    print("\n" + "="*60)
    print("BATCH PREDICTION ON MULTIPLE TURBINES")
    print("="*60)
    
    turbine_ids = ['0', '10', '13']
    
    for tid in turbine_ids:
        print(f"\n📊 Turbine {tid}:")
        try:
            preds, probs = predict_with_custom_threshold(
                data_path,
                tid,
                threshold=threshold
            )
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)
    print(f"\n💡 Visualizations saved to results/predictions_turbine_*_optimized.png")
    print(f"💡 To try different thresholds, modify the 'threshold' variable in main()")


if __name__ == "__main__":
    main()
