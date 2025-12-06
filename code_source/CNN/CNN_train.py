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

WINDOW_SIZE = 20

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
        # Note: X_frames is (N, 6). X_batch is (batch, 30, 6).
        # We want (batch, 6, 30, 1).
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
        # Valid start indices for this segment: 0 to length - window_size
        if length >= window_size:
            # relative indices
            rel_indices = np.arange(length - window_size + 1)
            # absolute indices
            abs_indices = rel_indices + current_offset
            indices.extend(abs_indices)
        current_offset += length
    return np.array(indices, dtype=np.int32)

def get_pos_weight(y_train):
    # y_train shape (N, 5)
    n_pos = np.sum(y_train)
    n_neg = np.size(y_train) - n_pos
    pos_weight = n_neg / (n_pos + 1e-7)
    return pos_weight

def weighted_bce_loss(pos_weight):
    def loss(y_true, y_pred):
        # y_true: (batch, 5), y_pred: (batch, 5) logits
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(y_true, tf.float32),
            logits=y_pred,
            pos_weight=pos_weight
        )
    return loss

def build_cnn_model(input_shape):
    # Input: (6, WINDOW_SIZE, 1)
    inputs = layers.Input(shape=input_shape)
    
    # Conv Block 1
    # Padding='valid' to reduce size
    x = layers.Conv2D(2, (1, 3), padding='valid', 
                      kernel_regularizer=regularizers.l2(0.001))(inputs) # (6, 28, 2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x) # (6, 14, 2)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 2
    x = layers.Conv2D(4, (1, 3), padding='valid', 
                      kernel_regularizer=regularizers.l2(0.001))(x) # (6, 12, 4)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((1, 2))(x) # (6, 6, 4)
    x = layers.Dropout(0.2)(x)
    
    # Flatten
    # Size: 6 * 6 * 4 = 144
    x = layers.Flatten()(x)
    
    # Dense layer to mix features
    # 144 * 16 = 2304 params
    x = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer (5 fingers)
    # Changed to None (logits) to work with weighted_cross_entropy_with_logits
    outputs = layers.Dense(5, activation=None)(x)
    
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

    # Calculate pos_weight from all training frames (good approximation)
    pos_weight = get_pos_weight(y_train_frames)#/5
    print(f"Calculated pos_weight: {pos_weight:.4f}")

    # Generators
    batch_size = 8
    train_gen = WindowGenerator(X_train_frames, y_train_frames, train_indices, batch_size, window_size=WINDOW_SIZE, shuffle=True)
    val_gen = WindowGenerator(X_val_frames, y_val_frames, val_indices, batch_size, window_size=WINDOW_SIZE, shuffle=False)

    # Build model
    input_shape = (6, WINDOW_SIZE, 1)
    model = build_cnn_model(input_shape)
    model.summary()

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=weighted_bce_loss(pos_weight),
                  metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    
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
    model_path = OUTPUT_DIR / "finger_press_cnn_v4.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('CNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    print("Loss curve saved.")

if __name__ == "__main__":
    main()
