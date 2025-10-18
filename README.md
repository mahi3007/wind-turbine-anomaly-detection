# 🌀 Wind Turbine Anomaly Detection Using Hierarchical Transformer-LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📄 Overview

An advanced **deep learning-based anomaly detection system** for wind turbine predictive maintenance using **SCADA (Supervisory Control and Data Acquisition)** data. This project implements a state-of-the-art **Hierarchical Transformer-LSTM architecture** that combines the power of attention mechanisms with temporal modeling to identify abnormal operational patterns and potential faults in wind turbines at an early stage.

The system achieves **97.72% accuracy** and **0.9835 ROC-AUC** with optimized threshold strategies that balance precision and recall based on operational requirements, enabling **predictive maintenance**, **reducing downtime**, and **improving operational efficiency** of wind farms.

---

## 🎯 Key Highlights

- **🏆 High Performance**: 97.72% accuracy, 0.9835 ROC-AUC score
- **🧠 Advanced Architecture**: Hierarchical Transformer-LSTM with multi-head attention
- **⚖️ Flexible Thresholds**: 4 pre-optimized strategies (best_f1, balanced, high_precision, original)
- **📊 Comprehensive Analysis**: Detailed EDA, visualization, and evaluation metrics
- **🔧 Production-Ready**: Complete prediction pipeline with batch processing support
- **📈 Optimized Performance**: Class balancing, weighted loss, learning rate scheduling

---

## 🏗️ Architecture

The model employs a **Hierarchical Transformer-LSTM** architecture:

1. **Feature Embedding Layer**: Projects input features to higher-dimensional space (64-dim)
2. **Positional Encoding**: Adds temporal position information to sequences
3. **Transformer Encoder Layers**: Multi-head self-attention (4 heads, 2 layers) for capturing complex patterns
4. **Bidirectional LSTM**: Temporal modeling with 2 layers (64 hidden units)
5. **Classification Head**: Fully connected layers with dropout for anomaly detection

**Key Features**:
- Multi-head attention mechanism for pattern recognition
- Residual connections and layer normalization
- Dropout regularization (30%) to prevent overfitting
- Xavier initialization for stable training
- NaN handling and gradient clipping

---

## 📊 Performance Metrics

### Model Performance
- **ROC-AUC**: 0.9835 (Excellent discrimination)
- **Training Accuracy**: 97.72%
- **Test Accuracy**: 97.72%

### Threshold Strategies

| Strategy | Threshold | Precision | Recall | F1 Score | Use Case |
|----------|-----------|-----------|--------|----------|----------|
| **best_f1** ⭐ | 0.9662 | 58% | 93% | 71% | Recommended for most cases |
| **balanced** | 0.9768 | 61% | 61% | 61% | Equal precision-recall |
| **high_precision** | 0.9805 | 80% | 22% | 35% | Minimize false alarms |
| **original** | 0.1796 | 48% | 100% | 65% | Safety-critical (catch all) |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wind-turbine-anomaly-detection.git
cd wind-turbine-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from predict_anomalies import AnomalyPredictor

# Initialize with optimized threshold (recommended)
predictor = AnomalyPredictor(threshold_strategy='best_f1')

# Make predictions
predictions, probabilities, data, indices = predictor.predict_from_file(
    data_path='dataset_dl/Wind Farm A/datasets',
    turbine_id='0'
)

# Visualize results
predictor.visualize_predictions(predictions, probabilities, indices, data)
```

### Training the Model

```bash
# Train with default parameters
python improved_transformer_lstm.py

# The model will:
# - Load and preprocess SCADA data
# - Apply class balancing and data augmentation
# - Train with learning rate scheduling
# - Generate evaluation metrics and visualizations
# - Save best model to outputs/best_model.pth
```

---

## 📁 Project Structure

```
wind-turbine-anomaly-detection/
├── improved_transformer_lstm.py    # Main model architecture and training
├── predict_anomalies.py            # Prediction pipeline with threshold strategies
├── optimize_threshold.py           # Threshold optimization and analysis
├── predict_with_optimized_threshold.py  # Simplified prediction script
├── Eda.ipynb                       # Exploratory Data Analysis notebook
├── PREDICTION_GUIDE.md             # Comprehensive prediction documentation
├── dataset_dl/                     # Wind farm SCADA datasets
│   ├── Wind Farm A/
│   ├── Wind Farm B/
│   └── Wind Farm C/
├── results/                        # Model outputs and visualizations
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_analysis.png
│   └── predictions_turbine_*.png
└── outputs/
    └── best_model.pth              # Trained model weights
```

---

## 🔧 Features

### Data Processing
- **Multi-farm support**: Processes data from Wind Farms A, B, and C
- **Sequence generation**: Sliding window approach (128 timesteps, stride 32)
- **Feature scaling**: StandardScaler normalization
- **Class balancing**: RandomOverSampler for imbalanced datasets
- **Missing value handling**: Robust preprocessing pipeline

### Model Training
- **Weighted loss function**: BCELoss with class weights
- **Learning rate scheduling**: ReduceLROnPlateau for adaptive learning
- **Early stopping**: Prevents overfitting
- **Gradient clipping**: Ensures training stability
- **Comprehensive logging**: Training progress and metrics

### Evaluation & Visualization
- **ROC and PR curves**: Model discrimination analysis
- **Confusion matrix**: Classification performance
- **Threshold analysis**: Precision-recall trade-offs
- **Anomaly timeline**: Temporal visualization of predictions
- **Probability distributions**: Confidence analysis

---

## 📈 Dataset

The project uses **Wind Turbine SCADA Data for Early Fault Detection** containing:
- **3 Wind Farms** (A, B, C)
- **Multiple turbines** per farm
- **High-frequency sensor data**: Power output, wind speed, temperature, vibration, etc.
- **Labeled anomaly events**: Start/end timestamps for fault periods
- **Feature descriptions**: Detailed sensor information

**Data Format**:
- Time-series SCADA measurements (10-minute intervals)
- Event labels (normal/anomaly)
- Asset IDs and timestamps

---

## 🛠️ Advanced Usage

### Threshold Optimization

```bash
# Analyze and optimize detection thresholds
python optimize_threshold.py

# Outputs:
# - Precision-recall curves
# - Threshold comparison plots
# - Detailed metrics for each strategy
```

### Batch Processing

```python
predictor = AnomalyPredictor(threshold_strategy='best_f1')

turbine_ids = ['0', '10', '13', '14', '15']
results = {}

for tid in turbine_ids:
    preds, probs, _, _ = predictor.predict_from_file(
        data_path='dataset_dl/Wind Farm A/datasets',
        turbine_id=tid
    )
    results[tid] = {
        'anomalies': preds.sum(),
        'anomaly_rate': preds.sum() / len(preds) * 100
    }
```

### Custom Threshold

```python
# Use custom threshold for specific requirements
predictor = AnomalyPredictor(custom_threshold=0.85)
predictions, probabilities, _, _ = predictor.predict_from_file(...)
```

---

## 📊 Exploratory Data Analysis

The `Eda.ipynb` notebook provides comprehensive analysis:
- **Data structure** and sensor information
- **Time series characteristics** and trends
- **Anomaly event patterns** across turbines
- **Feature correlations** and distributions
- **Missing value analysis** and data quality assessment

---

## 🧠 Technologies & Dependencies

- **Python 3.8+**
- **PyTorch 2.0+**: Deep learning framework
- **NumPy, Pandas**: Data manipulation
- **Scikit-learn**: Preprocessing and metrics
- **Imbalanced-learn**: Class balancing
- **Matplotlib, Seaborn**: Visualization
- **tqdm**: Progress tracking

---

## 📖 Documentation

- **[PREDICTION_GUIDE.md](PREDICTION_GUIDE.md)**: Comprehensive guide for making predictions
  - Threshold strategy selection
  - Performance comparison
  - Advanced usage examples
  - Decision-making guide

---

## 🎯 Use Cases

### Safety-Critical Applications
→ Use **`original`** threshold (100% recall) to catch all anomalies

### Balanced Operations (Maintenance Planning)
→ Use **`best_f1`** threshold (recommended) for optimal performance

### Cost-Sensitive Operations (Expensive Inspections)
→ Use **`high_precision`** threshold to minimize false alarms

### Research & Analysis
→ Use **`balanced`** threshold for equal precision-recall

---

## 🚀 Future Enhancements

- [ ] Real-time streaming data processing
- [ ] Multi-turbine correlation analysis
- [ ] Interactive dashboard for turbine health monitoring
- [ ] Model interpretability with attention visualization
- [ ] Deployment as REST API service
- [ ] Integration with SCADA systems
- [ ] Ensemble methods for improved robustness
- [ ] Transfer learning across wind farms

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{wind_turbine_anomaly_detection,
  title={Wind Turbine Anomaly Detection Using Hierarchical Transformer-LSTM},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/wind-turbine-anomaly-detection}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue or contact [your.email@example.com](mailto:your.email@example.com)

---

## 🙏 Acknowledgments

- Wind Turbine SCADA dataset providers
- PyTorch and scikit-learn communities
- Research papers on transformer architectures and anomaly detection

---

**Last Updated**: October 2025  
**Model Version**: hierarchical_transformer_lstm_v2.pth  
**Status**: Production-ready
