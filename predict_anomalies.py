"""
Prediction Script for Hierarchical Transformer-LSTM Wind Turbine Anomaly Detection

This script loads the trained model and makes predictions on new data.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import model architecture from training script
from improved_transformer_lstm import HierarchicalTransformerLSTM, WindTurbineDataset
from torch.utils.data import DataLoader

class AnomalyPredictor:
    """Class for loading trained model and making predictions"""
    
    def __init__(self, model_path='results/hierarchical_transformer_lstm_v2.pth', 
                 custom_threshold=None, threshold_strategy='best_f1'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model checkpoint
            custom_threshold: Custom threshold value (overrides threshold_strategy)
            threshold_strategy: Threshold strategy to use
                - 'original': Use threshold from training (0.1796) - High recall, lower precision
                - 'best_f1': Use threshold for best F1 score (0.9662) - Balanced [DEFAULT]
                - 'balanced': Use threshold for balanced precision-recall (0.9768)
                - 'high_precision': Use threshold for ≥80% precision (0.9805) - Conservative
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.config = checkpoint['config']
            self.scaler = checkpoint['scaler']
            self.original_threshold = checkpoint['threshold']
        except Exception as e:
            print(f"Warning: Could not load scaler from checkpoint: {e}")
            print("Will create new scaler during data preprocessing")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.config = checkpoint.get('config', {
                'seq_length': 128,
                'stride': 32,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.3
            })
            self.scaler = None
            self.original_threshold = checkpoint.get('threshold', 0.1796)
        
        # Define threshold strategies based on optimization results
        self.threshold_strategies = {
            'original': 0.1796,      # Original from training - 100% recall, 48% precision
            'best_f1': 0.9662,       # Best F1 score - 93% recall, 58% precision [RECOMMENDED]
            'balanced': 0.9768,      # Balanced precision-recall - 61% both
            'high_precision': 0.9805 # High precision ≥80% - 22% recall, 80% precision
        }
        
        # Set threshold based on strategy or custom value
        if custom_threshold is not None:
            self.threshold = custom_threshold
            self.strategy_used = f'custom ({custom_threshold:.4f})'
        elif threshold_strategy in self.threshold_strategies:
            self.threshold = self.threshold_strategies[threshold_strategy]
            self.strategy_used = threshold_strategy
        else:
            print(f"Warning: Unknown strategy '{threshold_strategy}'. Using 'best_f1'.")
            self.threshold = self.threshold_strategies['best_f1']
            self.strategy_used = 'best_f1'
        
        print(f"Original training threshold: {self.original_threshold:.4f}")
        print(f"Using threshold: {self.threshold:.4f} (strategy: {self.strategy_used})")
        
        # Print expected performance
        if self.strategy_used == 'best_f1':
            print("  Expected: ~93% recall, ~58% precision, ~71% F1 [BALANCED]")
        elif self.strategy_used == 'balanced':
            print("  Expected: ~61% recall, ~61% precision [EQUAL]")
        elif self.strategy_used == 'high_precision':
            print("  Expected: ~22% recall, ~80% precision [CONSERVATIVE]")
        elif self.strategy_used == 'original':
            print("  Expected: ~100% recall, ~48% precision [CATCH ALL]")
        
        # Initialize model architecture
        # Note: input_dim will be set when loading data
        self.model = None
        self.model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    def load_model(self, input_dim):
        """Load model with specified input dimension"""
        self.model = HierarchicalTransformerLSTM(
            input_dim=input_dim,
            seq_len=self.config['seq_length'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.model_state_dict)
        self.model.eval()
        print("Model loaded successfully!")
    
    def preprocess_data(self, data_path, turbine_id=None):
        """
        Preprocess data for prediction
        
        Args:
            data_path: Path to CSV file or directory containing turbine data
            turbine_id: Specific turbine ID to process (optional)
        
        Returns:
            sequences: Preprocessed sequences ready for prediction
            original_data: Original data for reference
        """
        print(f"Loading data from {data_path}...")
        
        # Load data
        if os.path.isfile(data_path):
            data = pd.read_csv(data_path, sep=';')
        else:
            # Load from directory
            if turbine_id is not None:
                file_path = os.path.join(data_path, f'{turbine_id}.csv')
                data = pd.read_csv(file_path, sep=';')
            else:
                raise ValueError("Please specify turbine_id when providing a directory path")
        
        # Store original data
        original_data = data.copy()
        
        # Drop non-numeric columns
        cols_to_drop = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']
        data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
        
        # Remove constant features
        data = data.loc[:, (data != data.iloc[0]).any()]
        
        # Normalize using the same scaler from training
        if self.scaler is not None:
            normalized_data = self.scaler.transform(data)
        else:
            # Create and fit new scaler if not available
            print("Warning: Using new scaler (not from training). Results may vary.")
            self.scaler = StandardScaler()
            normalized_data = self.scaler.fit_transform(data)
        
        # Create sequences
        seq_length = self.config['seq_length']
        stride = self.config['stride']
        
        sequences = []
        sequence_indices = []
        
        for i in range(0, len(normalized_data) - seq_length + 1, stride):
            seq = normalized_data[i:i + seq_length]
            sequences.append(seq)
            sequence_indices.append((i, i + seq_length))
        
        sequences = np.array(sequences)
        
        print(f"Created {len(sequences)} sequences from {len(data)} data points")
        
        return sequences, original_data, sequence_indices
    
    def predict(self, sequences, batch_size=64):
        """
        Make predictions on sequences
        
        Args:
            sequences: Preprocessed sequences
            batch_size: Batch size for prediction
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create dataset and dataloader
        # Create dummy labels (not used for prediction)
        dummy_labels = np.zeros(len(sequences))
        dataset = WindTurbineDataset(sequences, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        
        print("Making predictions...")
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                all_probs.extend(outputs.cpu().numpy())
        
        probabilities = np.array(all_probs)
        predictions = (probabilities >= self.threshold).astype(int)
        
        print(f"Predictions complete!")
        print(f"Anomaly rate: {predictions.sum() / len(predictions) * 100:.2f}%")
        
        return predictions, probabilities
    
    def predict_from_file(self, data_path, turbine_id=None, batch_size=64):
        """
        End-to-end prediction from file
        
        Args:
            data_path: Path to data file or directory
            turbine_id: Turbine ID (if directory provided)
            batch_size: Batch size for prediction
        
        Returns:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            original_data: Original data
            sequence_indices: Indices of sequences in original data
        """
        # Preprocess data
        sequences, original_data, sequence_indices = self.preprocess_data(
            data_path, turbine_id
        )
        
        # Load model if not already loaded
        if self.model is None:
            input_dim = sequences.shape[2]
            self.load_model(input_dim)
        
        # Make predictions
        predictions, probabilities = self.predict(sequences, batch_size)
        
        return predictions, probabilities, original_data, sequence_indices
    
    def visualize_predictions(self, predictions, probabilities, sequence_indices, 
                            original_data, save_path='results/predictions.png'):
        """
        Visualize predictions over time
        
        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            sequence_indices: Indices of sequences
            original_data: Original data
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Anomaly predictions over time
        time_points = [idx[0] for idx in sequence_indices]
        axes[0].scatter(time_points, predictions, c=predictions, 
                       cmap='RdYlGn_r', alpha=0.6, s=20)
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Prediction (0=Normal, 1=Anomaly)')
        axes[0].set_title('Anomaly Predictions Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction probabilities
        axes[1].plot(time_points, probabilities, color='blue', alpha=0.7, linewidth=1)
        axes[1].axhline(y=self.threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.threshold:.3f})')
        axes[1].fill_between(time_points, self.threshold, probabilities, 
                            where=(probabilities >= self.threshold),
                            alpha=0.3, color='red', label='Anomaly')
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Anomaly Probability')
        axes[1].set_title('Anomaly Probability Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Histogram of probabilities
        axes[2].hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[2].axvline(x=self.threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.threshold:.3f})')
        axes[2].set_xlabel('Anomaly Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Anomaly Probabilities')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()
    
    def evaluate_predictions(self, predictions, true_labels):
        """
        Evaluate predictions against true labels
        
        Args:
            predictions: Binary predictions
            true_labels: True labels
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0)
        }
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("="*60)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                   target_names=['Normal', 'Anomaly']))
        
        return metrics


def main():
    """Example usage of the predictor"""
    
    print("="*60)
    print("WIND TURBINE ANOMALY DETECTION - PREDICTION")
    print("="*60)
    print("\n📊 Available Threshold Strategies:")
    print("  1. 'best_f1' (DEFAULT): Best F1 score - 93% recall, 58% precision")
    print("  2. 'balanced': Equal precision-recall - 61% both")
    print("  3. 'high_precision': Conservative - 22% recall, 80% precision")
    print("  4. 'original': Catch all anomalies - 100% recall, 48% precision")
    print("  5. custom_threshold=X.XX: Use your own threshold value")
    
    # Initialize predictor with optimized threshold (best_f1 by default)
    predictor = AnomalyPredictor(
        model_path='results/hierarchical_transformer_lstm_v2.pth',
        threshold_strategy='best_f1'  # Change this to 'balanced', 'high_precision', or 'original'
        # Or use: custom_threshold=0.5  # for a custom value
    )
    
    # Example 1: Predict on a single turbine file
    print("\n" + "="*60)
    print("EXAMPLE 1: Predict on single turbine")
    print("="*60)
    
    data_path = 'dataset_dl/Wind Farm A/datasets'
    turbine_id = '0'  # Change this to predict on different turbines
    
    predictions, probabilities, original_data, sequence_indices = predictor.predict_from_file(
        data_path=data_path,
        turbine_id=turbine_id,
        batch_size=64
    )
    
    # Visualize predictions
    predictor.visualize_predictions(
        predictions, 
        probabilities, 
        sequence_indices,
        original_data,
        save_path=f'results/predictions_turbine_{turbine_id}.png'
    )
    
    # Example 2: Evaluate predictions if true labels are available
    print("\n" + "="*60)
    print("EXAMPLE 2: Evaluate predictions with true labels")
    print("="*60)
    
    # Load event info to get true labels
    event_info = pd.read_csv('dataset_dl/Wind Farm A/event_info.csv', sep=';')
    turbine_events = event_info[event_info['asset'] == int(turbine_id)]
    
    # Create true labels for sequences
    true_labels = np.zeros(len(predictions))
    for _, event in turbine_events.iterrows():
        if event['event_label'] == 'anomaly':
            start_idx = event['event_start_id']
            end_idx = event['event_end_id']
            # Mark sequences that overlap with anomaly period
            for i, (seq_start, seq_end) in enumerate(sequence_indices):
                if not (seq_end < start_idx or seq_start > end_idx):
                    true_labels[i] = 1
    
    # Evaluate
    metrics = predictor.evaluate_predictions(predictions, true_labels)
    
    # Example 3: Batch prediction on multiple turbines
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch prediction on multiple turbines")
    print("="*60)
    
    turbine_ids = ['0', '10', '13']  # Add more turbine IDs as needed
    
    print(f"\n{'Turbine ID':<12} {'Sequences':<12} {'Anomaly Rate':<15} {'Anomalies Detected':<20}")
    print("-" * 60)
    
    for tid in turbine_ids:
        try:
            preds, probs, _, _ = predictor.predict_from_file(
                data_path=data_path,
                turbine_id=tid,
                batch_size=64
            )
            anomaly_rate = preds.sum() / len(preds) * 100
            anomaly_count = preds.sum()
            print(f"{tid:<12} {len(preds):<12} {anomaly_rate:>6.2f}%{'':<8} {anomaly_count:<20}")
        except Exception as e:
            print(f"{tid:<12} Error: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ PREDICTION COMPLETE!")
    print("="*60)
    print(f"\n💡 Tips:")
    print(f"  - Visualizations saved to: results/predictions_turbine_*.png")
    print(f"  - Current threshold: {predictor.threshold:.4f} ({predictor.strategy_used})")
    print(f"  - To change threshold, modify 'threshold_strategy' parameter")
    print(f"  - Run optimize_threshold.py to analyze different thresholds")
    print("="*60)


if __name__ == "__main__":
    main()
