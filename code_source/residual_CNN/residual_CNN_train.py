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
            
        # Reshape X for CNN: (batch, 30, 6) -> (batch, 6, 30, 1)
        X_batch_np = np.array(X_batch) # (batch, 30, 6)
        X_batch_np = np.transpose(X_batch_np, (0, 2, 1)) # (batch, 6, 30)
        X_batch_np = X_batch_np[..., np.newaxis] # (batch, 6, 30, 1)
        
        return X_batch_np, np.array(y_batch)

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

def build_residual_cnn_model(input_shape):
    # Input: (6, 30, 1)
    inputs = layers.Input(shape=input_shape)
    
    # --- Convolutional Branch ---
    # Goal: Reduce time dimension from 30 to 3, keep sensors=6
    
    # Block 1: 10 -> 8 -> 4
    x = layers.Conv2D(2, (1, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001))(inputs) # (6, 8, 2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x) # (6, 4, 2)
    
    # Block 2: 4 -> 2 -> 1
    x = layers.Conv2D(4, (1, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x) # (6, 2, 4)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x) # (6, 1, 4)
    
    # Flatten Conv output
    # Shape: 6 * 1 * 4 = 24
    conv_flat = layers.Flatten()(x)
    
    # --- Skip Connection Branch ---
    # Take the last 1 frame of the input (Current state)
    # Input is (batch, 6, 10, 1)
    # We want (batch, 6, 1, 1)
    # Slicing: [:, :, -1:, :]
    skip = layers.Lambda(lambda t: t[:, :, -1:, :])(inputs)
    skip_flat = layers.Flatten()(skip) # 6 * 1 * 1 = 6
    
    # --- Concatenation ---
    # 24 + 6 = 30 features
    merged = layers.Concatenate()([conv_flat, skip_flat])
    
    # --- Dense Layers ---
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(merged)
    x = layers.Dropout(0.2)(x)
    
    # Output layer (5 fingers) - Sigmoid for 0-1 probability
    outputs = layers.Dense(5, activation='sigmoid')(x)
    
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
    input_shape = (6, WINDOW_SIZE, 1)
    model = build_residual_cnn_model(input_shape)
    model.summary()

    # Compile
    # Using standard binary_crossentropy because we use sigmoid activation
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
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
    model_path = OUTPUT_DIR / "residual_cnn.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Residual CNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    print("Loss curve saved.")

if __name__ == "__main__":
    main()
