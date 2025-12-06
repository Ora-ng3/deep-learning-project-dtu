import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
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
    model = build_fcnn_model(input_shape)
    model.summary()

    # Compile
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save model
    model_path = OUTPUT_DIR / "fcnn_model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('FCNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "fcnn_loss_curve.png")
    print("Loss curve saved.")

if __name__ == "__main__":
    main()
