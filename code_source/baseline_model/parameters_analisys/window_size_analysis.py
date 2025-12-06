import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
DATASET_PATH = CURRENT_DIR.parent / "data" / "dataset" / "dataset_scale_25k.npz"
OUTPUT_DIR = CURRENT_DIR

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_frames, y_frames, indices, batch_size, window_size, shuffle=True):
        self.X_frames = X_frames
        self.y_frames = y_frames
        self.indices = indices
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        
        X_batch = []
        y_batch = []
        
        for i in batch_indices:
            # X window: i to i+window_size
            X_batch.append(self.X_frames[i : i+self.window_size])
            # y label: last frame of the window -> i+window_size-1
            y_batch.append(self.y_frames[i + self.window_size - 1])
            
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    data = np.load(path)
    return data

def compute_indices(lengths, window_size):
    indices = []
    current_offset = 0
    for length in lengths:
        # Valid start indices for this segment: 0 to length - window_size
        if length >= window_size:
            # relative indices
            rel_indices = np.arange(length - window_size + 1)
            # absolute indices
            abs_indices = rel_indices + current_offset
            indices.extend(abs_indices)
        current_offset += length
    return np.array(indices, dtype=np.int32)

def build_fcnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(5, activation='sigmoid')
    ])
    return model

def main():
    # Load data
    print(f"Loading data from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    
    X_train_frames = data['X_train']
    y_train_frames = data['y_train']
    train_lengths = data['train_lengths']
    
    X_val_frames = data['X_val']
    y_val_frames = data['y_val']
    val_lengths = data['val_lengths']
    
    print(f"Data loaded.")
    
    results = []
    
    # Loop from 1 to 50
    for w in range(1, 51):
        print(f"Training with window size: {w}")
        
        # Compute indices
        train_indices = compute_indices(train_lengths, w)
        val_indices = compute_indices(val_lengths, w)
        
        if len(train_indices) == 0 or len(val_indices) == 0:
            print(f"Skipping window size {w} due to insufficient data.")
            continue

        # Generators
        batch_size = 32
        train_gen = WindowGenerator(X_train_frames, y_train_frames, train_indices, batch_size, window_size=w, shuffle=True)
        val_gen = WindowGenerator(X_val_frames, y_val_frames, val_indices, batch_size, window_size=w, shuffle=False)

        # Build model
        input_shape = (w, 6)
        model = build_fcnn_model(input_shape)

        # Compile
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, # Reduced patience for speed
            restore_best_weights=True
        )
        
        # Train
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50, # Reduced epochs for speed
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        # We need to predict on the validation set to calculate TP, FP, FN
        # Using the generator to get batches
        
        tp = 0
        fp = 0
        fn = 0
        
        # Iterate through validation generator
        for i in range(len(val_gen)):
            X_batch, y_batch = val_gen[i]
            preds = model.predict(X_batch, verbose=0)
            
            # Threshold
            preds_binary = (preds > 0.5).astype(int)
            y_true = y_batch.astype(int)
            
            # Calculate metrics for this batch (global across all 5 fingers)
            tp += np.sum((preds_binary == 1) & (y_true == 1))
            fp += np.sum((preds_binary == 1) & (y_true == 0))
            fn += np.sum((preds_binary == 0) & (y_true == 1))
            
        if (tp + fp + fn) > 0:
            proportion = tp / (tp + fp + fn)
        else:
            proportion = 0
            
        print(f"Window size {w} trained. True positives proportion: {proportion:.4f}")
        
        results.append({'window_size': w, 'proportion': proportion})
        
    # Save results
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "window_size_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['window_size'], df['proportion'], marker='o')
    plt.title('True Positive Proportion vs Window Size (FCNN)')
    plt.xlabel('Window Size')
    plt.ylabel('TP / (TP + FP + FN)')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "window_size_analysis.png")
    print("Plot saved.")

if __name__ == "__main__":
    main()
