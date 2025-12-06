import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
from pathlib import Path
import pandas as pd

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
DATASET_PATH = CURRENT_DIR.parent / "data" / "dataset" / "dataset_scale_25k.npz"

# Window sizes to check
TARGET_WINDOWS = [10, 15, 20, 30, 35]
N_RUNS = 5

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
            X_batch.append(self.X_frames[i : i+self.window_size])
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
        if length >= window_size:
            rel_indices = np.arange(length - window_size + 1)
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
    print(f"Loading data...")
    data = load_data(DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    train_lengths = data['train_lengths']
    X_val = data['X_val']
    y_val = data['y_val']
    val_lengths = data['val_lengths']
    
    results = []

    print(f"Starting stability check for windows: {TARGET_WINDOWS}")
    print(f"Running {N_RUNS} times for each window size.")

    for w in TARGET_WINDOWS:
        scores = []
        print(f"\n--- Window Size {w} ---")
        
        # Indices depend on window size
        train_indices = compute_indices(train_lengths, w)
        val_indices = compute_indices(val_lengths, w)
        
        for run in range(N_RUNS):
            # Generators
            train_gen = WindowGenerator(X_train, y_train, train_indices, 32, window_size=w, shuffle=True)
            val_gen = WindowGenerator(X_val, y_val, val_indices, 32, window_size=w, shuffle=False)
            
            # Model
            model = build_fcnn_model((w, 6))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train (fast)
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stopping], verbose=0)
            
            # Evaluate
            tp, fp, fn = 0, 0, 0
            for i in range(len(val_gen)):
                X_b, y_b = val_gen[i]
                preds = (model.predict(X_b, verbose=0) > 0.5).astype(int)
                y_true = y_b.astype(int)
                tp += np.sum((preds == 1) & (y_true == 1))
                fp += np.sum((preds == 1) & (y_true == 0))
                fn += np.sum((preds == 0) & (y_true == 1))
            
            denom = tp + fp + fn
            score = tp / denom if denom > 0 else 0
            scores.append(score)
            print(f"  Run {run+1}: {score:.4f}")
            
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"Window {w} Result: Mean = {mean_score:.4f}, Std = {std_score:.4f}")
        results.append({'window_size': w, 'mean': mean_score, 'std': std_score, 'runs': scores})

    # Summary
    print("\n=== FINAL SUMMARY ===")
    df = pd.DataFrame(results)
    print(df[['window_size', 'mean', 'std']])

if __name__ == "__main__":
    main()
