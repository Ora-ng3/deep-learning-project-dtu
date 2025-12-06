import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import models
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import seaborn as sns

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
DATASET_PATH = CURRENT_DIR.parent / "data" / "dataset" / "dataset_scale_25k.npz"
MODEL_PATH = CURRENT_DIR / "finger_press_cnn_v4.keras"
OUTPUT_DIR = CURRENT_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 20

THRESHOLD = 0.83
TOLERANCE = 1

FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

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

def get_window(X_frames, y_frames, index, window_size=WINDOW_SIZE):
    X = X_frames[index : index+window_size]
    y_seq = y_frames[index : index+window_size]
    y_last = y_frames[index + window_size - 1]
    return X, y_seq, y_last

def sigmoid(x):
    return tf.nn.sigmoid(x).numpy()

def plot_worst_windows(X_val_frames, y_val_frames, val_indices, model, fp_indices, fn_indices, window_size=WINDOW_SIZE):
    """
    Plots 5 FP and 5 FN examples.
    Shows the window used for prediction (0 to W) and the following window (W to 2W).
    """
    
    # Helper to plot a list of indices
    def plot_indices(indices, title_prefix):
        for i, idx in enumerate(indices[:5]): # Take first 5
            # We want to show [idx : idx + 2*window_size]
            # Check bounds
            if idx + 2*window_size > len(X_val_frames):
                continue
                
            X_context = X_val_frames[idx : idx + 2*window_size]
            y_context = y_val_frames[idx : idx + 2*window_size]
            
            # Prediction point is at idx + window_size - 1 (relative to X_val_frames)
            # Relative to X_context, it is at index window_size - 1
            pred_idx_rel = window_size - 1
            
            # Get prediction for this specific window
            X_input = X_val_frames[idx : idx + window_size]
            # Reshape for CNN: (window_size, 6) -> (6, window_size, 1)
            X_input_cnn = X_input.T[..., np.newaxis]
            X_input_batch = np.expand_dims(X_input_cnn, axis=0)
            
            pred_logits = model.predict(X_input_batch, verbose=0)[0]
            pred_prob = sigmoid(pred_logits)
            pred_class = (pred_prob > THRESHOLD).astype(int)
            
            # True label at prediction point
            y_true_at_pred = y_val_frames[idx + window_size - 1]
            
            # Plot
            fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
            frames = np.arange(len(X_context))
            
            X_T = X_context.T
            
            for f, ax in enumerate(axes):
                # Intensity
                ax.plot(frames, X_T[f], label='Intensity', color='blue', alpha=0.6)
                
                # True Label Sequence
                ax.plot(frames, y_context.T[f], label='True Label', color='green', linestyle='--', linewidth=2)
                
                # Prediction Point Marker
                ax.axvline(x=pred_idx_rel, color='gray', linestyle=':', alpha=0.5)
                
                # True Label at Pred Point
                ax.scatter(pred_idx_rel, y_true_at_pred[f], color='green', s=100, marker='o', label='True')
                
                # Predicted
                ax.scatter(pred_idx_rel, pred_prob[f], color='orange', s=100, marker='x', label='Pred Prob')
                ax.scatter(pred_idx_rel, pred_class[f], color='red', s=50, marker='.', label='Pred Class')
                
                ax.set_ylabel(FINGER_NAMES[f])
                ax.set_ylim(-0.1, 1.1)
                if f == 0:
                    ax.legend(loc='upper right')
            
            plt.suptitle(f"{title_prefix} Example {i+1} (Window Start: {idx})")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{title_prefix.lower().replace(' ', '_')}_{i}.png")
            plt.close()

    if len(fp_indices) > 0:
        print(f"Plotting {min(5, len(fp_indices))} False Positives...")
        plot_indices(fp_indices, "False Positive")
    
    if len(fn_indices) > 0:
        print(f"Plotting {min(5, len(fn_indices))} False Negatives...")
        plot_indices(fn_indices, "False Negative")

def main():
    # Load data
    print("Loading data...")
    data = load_data(DATASET_PATH)
    X_val_frames = data['X_val']
    y_val_frames = data['y_val']
    val_lengths = data['val_lengths']
    
    # Compute indices
    val_indices = compute_indices(val_lengths, WINDOW_SIZE)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = models.load_model(MODEL_PATH, compile=False)
    
    # Evaluate with Tolerance
    print(f"Evaluating with tolerance ({TOLERANCE} frames)...")
    
    tp_tol = 0
    fp_tol = 0
    fn_tol = 0
    tn_tol = 0
    
    fp_indices = [] # Store indices of False Positives
    fn_indices = [] # Store indices of False Negatives
    
    batch_size = 1024
    all_preds_prob = []
    all_indices = []
    
    start_time = time.time()
    
    num_batches = int(np.ceil(len(val_indices) / batch_size))
    for b in range(num_batches):
        batch_indices = val_indices[b*batch_size : (b+1)*batch_size]
        X_batch = []
        # y_batch = [] # Not needed for prediction loop
        for idx in batch_indices:
            X, _, _ = get_window(X_val_frames, y_val_frames, idx)
            # Reshape X for CNN: (30, 6) -> (6, 30, 1)
            X_cnn = X.T[..., np.newaxis]
            X_batch.append(X_cnn)
            
        X_batch = np.array(X_batch)
        
        logits = model.predict(X_batch, verbose=0)
        probs = sigmoid(logits)
        all_preds_prob.append(probs)
        all_indices.append(batch_indices)
        
    end_time = time.time()
    
    all_preds_prob = np.vstack(all_preds_prob)
    all_indices = np.concatenate(all_indices)
    
    inference_time_ms = (end_time - start_time) * 1000 / len(val_indices)
    print(f"Inference time per window: {inference_time_ms:.4f} ms")
    
    # 2. Calculate Metrics with Tolerance
    for i, idx in enumerate(all_indices):
        pred_probs = all_preds_prob[i]
        pred_classes = (pred_probs > THRESHOLD).astype(int)
        
        # True label at the end of the window
        target_idx = idx + WINDOW_SIZE - 1
        y_true = y_val_frames[target_idx]
        
        # Check each finger independently
        for f in range(5):
            p = pred_classes[f]
            t = y_true[f]
            
            if p == 1 and t == 1:
                tp_tol += 1
            elif p == 0 and t == 0:
                tn_tol += 1
            elif p == 1 and t == 0:
                # Potential FP. Check tolerance.
                start_check = max(0, target_idx - TOLERANCE)
                end_check = min(len(y_val_frames), target_idx + TOLERANCE + 1)
                
                nearby_labels = y_val_frames[start_check:end_check, f]
                if 1 in nearby_labels:
                    tp_tol += 1 # Tolerant TP
                else:
                    fp_tol += 1
                    if len(fp_indices) < 100:
                        fp_indices.append(idx)
                        
            elif p == 0 and t == 1:
                # Potential FN. Check tolerance.
                start_check = max(0, target_idx - TOLERANCE)
                end_check = min(len(y_val_frames), target_idx + TOLERANCE + 1)
                
                nearby_labels = y_val_frames[start_check:end_check, f]
                if 0 in nearby_labels:
                    tn_tol += 1 # Tolerant TN
                else:
                    fn_tol += 1
                    if len(fn_indices) < 100:
                        fn_indices.append(idx)

    # Plot Worst Windows
    fp_indices = sorted(list(set(fp_indices)))
    fn_indices = sorted(list(set(fn_indices)))
    
    plot_worst_windows(X_val_frames, y_val_frames, val_indices, model, fp_indices, fn_indices)
    
    # Use tolerant metrics for the final report
    tn, fp, fn, tp = tn_tol, fp_tol, fn_tol, tp_tol
    
    # Construct confusion matrix manually for plotting
    cm = np.array([[tn, fp], [fn, tp]])
    
    total_relevant = fp + fn + tp
    if total_relevant == 0: total_relevant = 1

    print(f"\n=== Confusion Matrix (Tolerance = {TOLERANCE} frame) ===")
    print(f"False Positives: {fp} ({fp/total_relevant*100:.2f}%)")
    print(f"False Negatives: {fn} ({fn/total_relevant*100:.2f}%)")
    print(f"True Positives: {tp} ({tp/total_relevant*100:.2f}%)")
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Save CM plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Pressed', 'Pressed'], 
                yticklabels=['Not Pressed', 'Pressed'])
    plt.title('Confusion Matrix (CNN) - Tolerant')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(OUTPUT_DIR / "confusion_matrix_cnn.png")
    print("Confusion matrix saved.")

if __name__ == "__main__":
    main()
