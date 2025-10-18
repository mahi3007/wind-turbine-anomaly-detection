# Wind Turbine Anomaly Detection - Prediction Guide

## 🎯 Quick Start

### Basic Usage
```python
from predict_anomalies import AnomalyPredictor

# Initialize with optimized threshold (best_f1 - RECOMMENDED)
predictor = AnomalyPredictor(threshold_strategy='best_f1')

# Make predictions
predictions, probabilities, data, indices = predictor.predict_from_file(
    data_path='dataset_dl/Wind Farm A/datasets',
    turbine_id='0'
)
```

### Run Examples
```bash
python predict_anomalies.py
```

---

## 📊 Threshold Strategies

The model now supports **4 pre-optimized threshold strategies** based on your precision-recall requirements:

### 1. **`best_f1`** (DEFAULT) ⭐ **RECOMMENDED**
- **Threshold**: 0.9662
- **Precision**: ~58%
- **Recall**: ~93%
- **F1 Score**: ~71%
- **Use when**: You want the best overall balance
- **Trade-off**: Minimal - best for most use cases

```python
predictor = AnomalyPredictor(threshold_strategy='best_f1')
```

### 2. **`balanced`**
- **Threshold**: 0.9768
- **Precision**: ~61%
- **Recall**: ~61%
- **F1 Score**: ~61%
- **Use when**: You want equal precision and recall
- **Trade-off**: Slightly lower recall than best_f1

```python
predictor = AnomalyPredictor(threshold_strategy='balanced')
```

### 3. **`high_precision`**
- **Threshold**: 0.9805
- **Precision**: ~80%
- **Recall**: ~22%
- **F1 Score**: ~35%
- **Use when**: False alarms are very costly, you only want high-confidence anomalies
- **Trade-off**: Will miss many anomalies (low recall)

```python
predictor = AnomalyPredictor(threshold_strategy='high_precision')
```

### 4. **`original`**
- **Threshold**: 0.1796
- **Precision**: ~48%
- **Recall**: ~100%
- **F1 Score**: ~65%
- **Use when**: You cannot afford to miss any anomalies (safety-critical)
- **Trade-off**: Many false alarms (low precision)

```python
predictor = AnomalyPredictor(threshold_strategy='original')
```

### 5. **Custom Threshold**
- **Use when**: You have specific requirements
- **How**: Set any value between 0 and 1

```python
predictor = AnomalyPredictor(custom_threshold=0.75)
```

---

## 📈 Performance Comparison

| Strategy | Threshold | Precision | Recall | F1 Score | False Positives | False Negatives |
|----------|-----------|-----------|--------|----------|-----------------|-----------------|
| **best_f1** ⭐ | 0.9662 | **58%** | **93%** | **71%** | 49 | 5 |
| balanced | 0.9768 | 61% | 61% | 61% | 42 | 28 |
| high_precision | 0.9805 | 80% | 22% | 35% | 18 | 56 |
| original | 0.1796 | 48% | 100% | 65% | 78 | 0 |

*Based on Turbine 0 evaluation*

---

## 🔧 Advanced Usage

### Example 1: Compare Multiple Strategies
```python
strategies = ['original', 'best_f1', 'balanced', 'high_precision']

for strategy in strategies:
    predictor = AnomalyPredictor(threshold_strategy=strategy)
    predictions, probs, _, _ = predictor.predict_from_file(
        data_path='dataset_dl/Wind Farm A/datasets',
        turbine_id='0'
    )
    print(f"{strategy}: {predictions.sum()} anomalies detected")
```

### Example 2: Batch Processing with Custom Threshold
```python
predictor = AnomalyPredictor(custom_threshold=0.8)

turbine_ids = ['0', '10', '13', '14', '15']
results = {}

for tid in turbine_ids:
    preds, probs, _, _ = predictor.predict_from_file(
        data_path='dataset_dl/Wind Farm A/datasets',
        turbine_id=tid
    )
    results[tid] = {
        'total_sequences': len(preds),
        'anomalies': preds.sum(),
        'anomaly_rate': preds.sum() / len(preds) * 100
    }
```

### Example 3: Export Predictions to CSV
```python
import pandas as pd

predictor = AnomalyPredictor(threshold_strategy='best_f1')
predictions, probabilities, original_data, sequence_indices = predictor.predict_from_file(
    data_path='dataset_dl/Wind Farm A/datasets',
    turbine_id='0'
)

# Create results dataframe
results_df = pd.DataFrame({
    'sequence_id': range(len(predictions)),
    'start_index': [idx[0] for idx in sequence_indices],
    'end_index': [idx[1] for idx in sequence_indices],
    'anomaly_probability': probabilities,
    'is_anomaly': predictions
})

# Save to CSV
results_df.to_csv('results/predictions_turbine_0.csv', index=False)
print("Predictions exported to CSV!")
```

---

## 🛠️ Optimization Tools

### Analyze Thresholds
Run the threshold optimization script to analyze different thresholds:

```bash
python optimize_threshold.py
```

This will:
- Generate precision-recall curves
- Compare different threshold strategies
- Show detailed metrics for each threshold
- Save analysis plots to `results/threshold_analysis.png`

### Modify Target Precision/Recall
Edit `optimize_threshold.py` to find thresholds for your specific targets:

```python
target_precision = 0.90  # Change to your desired precision
target_recall = 0.85     # Change to your desired recall
```

---

## 📁 Output Files

### Predictions
- `results/predictions_turbine_*.png` - Visualization of predictions over time
- Includes: anomaly timeline, probability curves, distributions

### Evaluation (if true labels available)
- Confusion matrix
- Classification report
- Precision, Recall, F1 scores

### Optimization Analysis
- `results/threshold_analysis.png` - Comprehensive threshold comparison
- Precision-recall curves
- Metrics vs threshold plots
- Probability distributions

---

## 💡 Decision Guide

**Choose your threshold strategy based on your use case:**

### Safety-Critical Applications (e.g., preventing catastrophic failures)
→ Use **`original`** (100% recall)
- Catches all anomalies
- Accept more false alarms

### Balanced Operations (e.g., maintenance planning)
→ Use **`best_f1`** (RECOMMENDED)
- Best overall performance
- Good balance of precision and recall

### Cost-Sensitive Operations (e.g., expensive inspections)
→ Use **`high_precision`** (80% precision)
- Only flag high-confidence anomalies
- Minimize false alarms

### Equal Importance (e.g., research/analysis)
→ Use **`balanced`** (equal precision-recall)
- No bias towards either metric

---

## 🎓 Understanding the Metrics

### Precision
- **What it means**: Of all flagged anomalies, how many are actually anomalies?
- **High precision**: Fewer false alarms, but might miss some anomalies
- **Low precision**: More false alarms, but catches more anomalies

### Recall
- **What it means**: Of all actual anomalies, how many did we catch?
- **High recall**: Catches most/all anomalies, but more false alarms
- **Low recall**: Misses many anomalies, but fewer false alarms

### F1 Score
- **What it means**: Harmonic mean of precision and recall
- **High F1**: Good balance between precision and recall
- **Use**: When you want overall performance metric

---

## 🚀 Model Performance Summary

### Training Results
- **ROC-AUC**: 0.9835 (Excellent!)
- **Training Accuracy**: 97.72%
- **Test Accuracy**: 97.72%

### With Optimized Threshold (best_f1)
- **Precision**: 57.76% (↑9.76% from original)
- **Recall**: 93.06%
- **F1 Score**: 71.28% (↑6.42% from original)
- **False Positives**: Reduced by 37%

---

## 📞 Support

For questions or issues:
1. Check the confusion matrix in evaluation output
2. Run `optimize_threshold.py` to analyze performance
3. Try different threshold strategies
4. Adjust custom threshold based on your requirements

---

## 📝 Quick Reference

```python
# Import
from predict_anomalies import AnomalyPredictor

# Initialize (choose one)
predictor = AnomalyPredictor(threshold_strategy='best_f1')      # Recommended
predictor = AnomalyPredictor(threshold_strategy='balanced')     # Equal P/R
predictor = AnomalyPredictor(threshold_strategy='high_precision') # Conservative
predictor = AnomalyPredictor(threshold_strategy='original')     # Catch all
predictor = AnomalyPredictor(custom_threshold=0.75)            # Custom

# Predict
predictions, probabilities, data, indices = predictor.predict_from_file(
    data_path='dataset_dl/Wind Farm A/datasets',
    turbine_id='0'
)

# Visualize
predictor.visualize_predictions(predictions, probabilities, indices, data)

# Evaluate (if labels available)
metrics = predictor.evaluate_predictions(predictions, true_labels)
```

---

**Last Updated**: 2025-10-16
**Model Version**: hierarchical_transformer_lstm_v2.pth
**Default Threshold**: 0.9662 (best_f1 strategy)
