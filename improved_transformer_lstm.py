"""
Improved Hierarchical Transformer-LSTM Model for Wind Turbine Anomaly Detection

This implementation includes:
- Better data preprocessing and augmentation
- Optimized model architecture with proper regularization
- Class balancing and weighted loss
- Learning rate scheduling
- Comprehensive evaluation metrics
- Visualization tools
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class WindTurbineDataset(Dataset):
    """Custom Dataset for wind turbine data with sequence generation"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with layer normalization and dropout"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class HierarchicalTransformerLSTM(nn.Module):
    """
    Hierarchical Transformer-LSTM model for anomaly detection
    
    Architecture:
    1. Feature embedding
    2. Positional encoding
    3. Transformer encoder layers
    4. LSTM for temporal modeling
    5. Classification head
    """
    def __init__(self, input_dim, seq_len, nhead=4, num_layers=2, dropout=0.3):
        super(HierarchicalTransformerLSTM, self).__init__()
        
        # Model parameters
        self.d_model = 64
        self.seq_len = seq_len
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=5000)
        
        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_feedforward=256,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful initialization to prevent NaN"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Use smaller initialization for better stability
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Project input to higher dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            # Check for NaN after each layer
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Check for NaN after LSTM
        if torch.isnan(lstm_out).any():
            lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        
        # Use last timestep output for classification
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(out)  # (batch_size, 1)
        
        # Final NaN check
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.5)
        
        return logits.squeeze(-1)

def load_and_preprocess_data(base_path, seq_length=128, stride=32, test_size=0.2, val_size=0.1):
    """
    Load and preprocess wind turbine data with improved handling
    """
    print("Loading and preprocessing data...")
    
    all_data = []
    all_labels = []
    
    # Load data from all wind farms
    for farm in ['Wind Farm A']:  # Focus on Wind Farm A initially
        farm_path = os.path.join(base_path, farm)
        event_info = pd.read_csv(os.path.join(farm_path, 'event_info.csv'), sep=';')
        datasets_path = os.path.join(farm_path, 'datasets')
        
        for file in os.listdir(datasets_path):
            if file.endswith('.csv') and not file.startswith('comma_'):
                turbine_id = file.split('.')[0]
                try:
                    data = pd.read_csv(os.path.join(datasets_path, file), sep=';')
                    
                    # Drop non-numeric columns (timestamp, metadata, etc.)
                    cols_to_drop = ['time_stamp', 'asset_id', 'id', 'train_test', 'status_type_id']
                    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
                    
                    # Get anomaly labels for this turbine
                    turbine_events = event_info[event_info['asset'] == int(turbine_id)]
                    labels = np.zeros(len(data))
                    
                    for _, event in turbine_events.iterrows():
                        # Only mark anomalies, not normal events
                        if event['event_label'] == 'anomaly':
                            start_idx = event['event_start_id']
                            end_idx = event['event_end_id']
                            labels[start_idx:end_idx+1] = 1
                    
                    all_data.append(data)
                    all_labels.append(labels)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid data files found in the specified directory.")
    
    # Concatenate all data
    data = pd.concat(all_data, axis=0).reset_index(drop=True)
    labels = np.concatenate(all_labels)
    
    # Remove constant features
    data = data.loc[:, (data != data.iloc[0]).any()]
    
    # Global normalization
    print("Applying global normalization...")
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Create sequences with sliding windows, ensuring at least one anomaly per sequence
    print("Creating sequences with sliding windows...")
    sequences = []
    sequence_labels = []
    
    for i in range(0, len(normalized_data) - seq_length + 1, stride):
        seq = normalized_data[i:i + seq_length]
        seq_label = np.max(labels[i:i + seq_length])
        
        # Only include sequences with at least one anomaly point
        if seq_label > 0 or np.random.random() < 0.1:  # Keep 10% of normal sequences
            sequences.append(seq)
            sequence_labels.append(seq_label)
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    # Balance the dataset
    print("Balancing dataset...")
    ros = RandomOverSampler(random_state=42)
    n_samples, seq_len, n_features = sequences.shape
    X_reshaped = sequences.reshape(n_samples, -1)
    X_balanced, y_balanced = ros.fit_resample(X_reshaped, sequence_labels)
    sequences_balanced = X_balanced.reshape(-1, seq_len, n_features)
    
    # Split into train, validation, and test sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_balanced, y_balanced, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=val_size/(1-test_size), 
        random_state=42, 
        stratify=y_train
    )
    
    # Compute class weights for loss function
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print("Data preprocessing completed.")
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
    print(f"Class distribution - Test: {np.bincount(y_test.astype(int))}")
    print(f"Label range - Train: [{y_train.min()}, {y_train.max()}]")
    print(f"Label unique values: {np.unique(y_train)}")
    
    return (
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test,
        class_weight_dict, 
        scaler
    )

def train_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc="Training")):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        
        # Clamp outputs to valid range for BCE loss
        outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
        
        # Calculate loss with class weights
        if class_weights is not None:
            weight = torch.tensor([class_weights[int(y.item())] for y in batch_y], 
                                device=device, dtype=torch.float32)
            loss = nn.functional.binary_cross_entropy(outputs, batch_y, weight=weight)
        else:
            loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics at optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Convert probabilities to binary predictions
    binary_preds = (all_preds >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': best_threshold
    }

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics at optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    # Convert probabilities to binary predictions
    binary_preds = (all_preds >= best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': best_threshold,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

def plot_metrics(train_metrics, val_metrics, metric_name, save_path=None):
    """Plot training and validation metrics"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_metrics) + 1), train_metrics, label=f'Training {metric_name}')
    plt.plot(range(1, len(val_metrics) + 1), val_metrics, label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{metric_name}.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

def plot_pr_curve(y_true, y_scores, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'pr_curve.png'))
    plt.close()

def main():
    # Configuration
    config = {
        'data_path': 'dataset_dl',
        'seq_length': 128,
        'stride': 32,
        'batch_size': 64,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'dropout': 0.3,
        'nhead': 4,
        'num_layers': 2,
        'patience': 10,
        'output_dir': 'results'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    (X_train, y_train, 
     X_val, y_val, 
     X_test, y_test, 
     class_weights, 
     scaler) = load_and_preprocess_data(
        config['data_path'],
        seq_length=config['seq_length'],
        stride=config['stride']
    )
    
    # Ensure labels are float32 and in [0, 1]
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"After conversion - y_train dtype: {y_train.dtype}, range: [{y_train.min()}, {y_train.max()}]")
    
    # Convert to PyTorch datasets
    train_dataset = WindTurbineDataset(X_train, y_train)
    val_dataset = WindTurbineDataset(X_val, y_val)
    test_dataset = WindTurbineDataset(X_test, y_test)
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model
    input_dim = X_train.shape[2]  # Number of features
    model = HierarchicalTransformerLSTM(
        input_dim=input_dim,
        seq_len=config['seq_length'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # Track metrics
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, criterion, 
            optimizer, device, class_weights
        )
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Track metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_aucs.append(train_metrics['roc_auc'])
        val_aucs.append(val_metrics['roc_auc'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train AUC: {train_metrics['roc_auc']:.4f}, Val AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model = model.state_dict()
            torch.save(best_model, os.path.join(config['output_dir'], 'best_model.pth'))
            patience_counter = 0
            
            # Save best threshold
            best_threshold = val_metrics['threshold']
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_model.pth')))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, best_threshold)
    
    # Print test metrics
    print("\nTest Set Metrics:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    
    # Plot training curves
    plot_metrics(train_losses, val_losses, 'loss', config['output_dir'])
    plot_metrics(train_aucs, val_aucs, 'auc', config['output_dir'])
    
    # Plot evaluation metrics
    plot_confusion_matrix(
        test_metrics['all_labels'], 
        (test_metrics['all_preds'] >= best_threshold).astype(int),
        config['output_dir']
    )
    
    plot_roc_curve(
        test_metrics['all_labels'],
        test_metrics['all_preds'],
        config['output_dir']
    )
    
    plot_pr_curve(
        test_metrics['all_labels'],
        test_metrics['all_preds'],
        config['output_dir']
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'config': config,
        'threshold': best_threshold
    }, os.path.join(config['output_dir'], 'hierarchical_transformer_lstm_v2.pth'))
    
    print(f"\nTraining complete. Model and results saved to {config['output_dir']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print("ERROR OCCURRED:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")
