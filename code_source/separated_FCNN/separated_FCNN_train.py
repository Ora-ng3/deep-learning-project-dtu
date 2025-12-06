import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
DATASET_PATH = CURRENT_DIR.parent / "data" / "dataset" / "dataset_scale_25k.npz"
OUTPUT_DIR = CURRENT_DIR

WINDOW_SIZE = 10

class WindowGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_frames, y_frames, indices, batch_size, window_size=WINDOW_SIZE, shuffle=True):
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

def build_separated_fcnn_model(input_shape):
    # Input shape: (WINDOW_SIZE, 6)
    inputs = layers.Input(shape=input_shape)
    
    # 1. Discard the 6th sensor (Palm) -> Keep first 5
    # Shape: (batch, WINDOW_SIZE, 5)
    x = layers.Lambda(lambda t: t[:, :, :5])(inputs)
    
    # 2. Process each finger separately
    finger_outputs = []
    
    for i in range(5):
        # Extract finger i: (batch, WINDOW_SIZE, 1)
        finger_input = layers.Lambda(lambda t: t[:, :, i:i+1])(x)
        
        # Flatten: (batch, WINDOW_SIZE)
        flat = layers.Flatten()(finger_input)
        
        # FCNN Branch
        # 10 -> 16 -> 16 -> 1
        branch = layers.Dense(16, activation='relu')(flat)
        branch = layers.Dense(16, activation='relu')(branch)
        out = layers.Dense(1, activation='sigmoid')(branch)
        
        finger_outputs.append(out)
        
    # 3. Concatenate outputs: (batch, 5)
    outputs = layers.Concatenate()(finger_outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
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
    
    # Compute indices
    train_indices = compute_indices(train_lengths, WINDOW_SIZE)
    val_indices = compute_indices(val_lengths, WINDOW_SIZE)
    
    print(f"Train frames: {len(X_train_frames)}, Valid windows: {len(train_indices)}")
    print(f"Val frames: {len(X_val_frames)}, Valid windows: {len(val_indices)}")

    # Generators
    batch_size = 32
    train_gen = WindowGenerator(X_train_frames, y_train_frames, train_indices, batch_size, window_size=WINDOW_SIZE, shuffle=True)
    val_gen = WindowGenerator(X_val_frames, y_val_frames, val_indices, batch_size, window_size=WINDOW_SIZE, shuffle=False)

    # Build model
    input_shape = (WINDOW_SIZE, 6)
    model = build_separated_fcnn_model(input_shape)
    model.summary()

    # Compile
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=OUTPUT_DIR / "separated_fcnn_model.keras",
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_cb = callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_loss'
    )

    # Train
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig(OUTPUT_DIR / "training_history.png")
    print("Training done.")

if __name__ == "__main__":
    main()
