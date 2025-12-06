"""
build_dataset.py

Builds a dataset from CSV files in `raw_recordings`.
- Finds all CSV files in `raw_recordings`
- For each file:
  - Trims start/end using label transitions
  - Splits into Train (80%) and Validation (20%) segments (first 80% frames -> train, last 20% -> val)
- Concatenates all train segments into X_train, y_train
- Concatenates all val segments into X_val, y_val
- Saves lengths of each segment to allow reconstruction of boundaries
- Saves to `dataset/` as .npz
"""
from pathlib import Path
import numpy as np
import pandas as pd
import random

FILE_NAME = "dataset_scale_25k.npz"
SCALE = 25000.0  # default scaling factor for intensities

TRAIN_FRAC = 0.8  # fraction of frames per file for training

RECORDINGS_DIR = Path(__file__).parent.resolve() / "raw_recordings"
DATASET_DIR = Path(__file__).parent.resolve() / "dataset"

# Column names expected by the CSVs (order matters)
COLS = [
    'timestamp',
    'thumb_intensity','index_intensity','middle_intensity','ring_intensity','pinky_intensity','palm_intensity',
    'thumb_pos','index_pos','middle_pos','ring_pos','pinky_pos','palm_pos',
    'label_thumb','label_index','label_middle','label_ring','label_pinky'
]
EXPECTED_COLS_LEN = len(COLS)


def trim_df(df, label_cols):
    """
    Trim DataFrame start and end according to rule:
    - find first fall (a 1->0 transition in any label column) and make the index of that 0 +10 the new start
    - find last rise (a 0->1 transition when scanning from the end) and make that index -10 the new end
    If such transitions can't be found, returns the original df.
    """
    labels = df[label_cols].astype(int).to_numpy()
    if labels.shape[0] < 2:
        return df

    # first fall: find index k where labels[k] < labels[k-1] for any column -> means 1->0
    comp = np.any(labels[1:] < labels[:-1], axis=1)
    if np.any(comp):
        first_fall = np.argmax(comp) + 1 + 10  # move 10 frames later
    else:
        first_fall = 0

    # last rise: scan reversed for any labels_rev[1:] > labels_rev[:-1]
    rev_comp = np.any(labels[::-1][1:] > labels[::-1][:-1], axis=1)
    if np.any(rev_comp):
        last_rise = len(labels) - (np.argmax(rev_comp)) - 10  # move 10 frames earlier
    else:
        last_rise = len(labels)

    # safety: ensure valid slice
    if first_fall >= last_rise:
        return df.iloc[0:0]  # empty

    return df.iloc[first_fall:last_rise].reset_index(drop=True)


def build_dataset(recordings_dir, output_dir, train_frac=0.8, seed=42):
    recordings_dir = Path(recordings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted([p for p in recordings_dir.glob('*.csv') if p.is_file()])
    print(f"Found {len(csv_paths)} csv files in {recordings_dir}")

    all_X_train = []
    all_y_train = []
    train_lengths = []

    all_X_val = []
    all_y_val = []
    val_lengths = []
    
    all_y_total = [] # Store all labels for reference

    for p in csv_paths:
        try:
            df = pd.read_csv(p, delimiter=';')
        except Exception as e:
            print(f"Skipping {p} (read error): {e}")
            continue

        if df.empty:
            continue

        if df.shape[1] != EXPECTED_COLS_LEN:
            # try to fix columns if count matches
            if df.shape[1] == EXPECTED_COLS_LEN:
                 df.columns = COLS
            else:
                print(f"Skipping {p} (unexpected column count {df.shape[1]})")
                continue

        # enforce column names if header missing or wrong
        if list(df.columns) != COLS:
            df.columns = COLS

        # trim
        trimmed = trim_df(df, ['label_thumb','label_index','label_middle','label_ring','label_pinky'])
        if trimmed.empty:
            print(f"Skipping {p} after trimming (empty)")
            continue

        # Extract data
        X = trimmed[['thumb_intensity','index_intensity','middle_intensity','ring_intensity','pinky_intensity','palm_intensity']].to_numpy().astype(np.float32)
        y = trimmed[['label_thumb','label_index','label_middle','label_ring','label_pinky']].to_numpy().astype(np.int64)
        
        all_y_total.append(y)
        
        # Split into Train/Val
        n_frames = len(X)
        n_train = int(n_frames * train_frac)
        
        X_t = X[:n_train]
        y_t = y[:n_train]
        X_v = X[n_train:]
        y_v = y[n_train:]
        
        if len(X_t) > 0:
            all_X_train.append(X_t)
            all_y_train.append(y_t)
            train_lengths.append(len(X_t))
            
        if len(X_v) > 0:
            all_X_val.append(X_v)
            all_y_val.append(y_v)
            val_lengths.append(len(X_v))
            
        print(f"Processed {p.name}: {n_frames} frames -> {len(X_t)} train, {len(X_v)} val")

    if not all_X_train:
        raise SystemExit("Not enough data to build dataset.")

    # Concatenate
    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    train_lengths = np.array(train_lengths, dtype=np.int32)
    
    if all_y_total:
        y_all = np.vstack(all_y_total)
    else:
        y_all = np.empty((0, 5), dtype=np.int64)
    
    if all_X_val:
        X_val = np.vstack(all_X_val)
        y_val = np.vstack(all_y_val)
        val_lengths = np.array(val_lengths, dtype=np.int32)
    else:
        X_val = np.empty((0, 6), dtype=np.float32)
        y_val = np.empty((0, 5), dtype=np.int64)
        val_lengths = np.array([], dtype=np.int32)

    # Scale Intensities
    X_train = X_train / SCALE
    X_val = X_val / SCALE

    # Save as .npz
    out_file = output_dir / FILE_NAME
    np.savez_compressed(out_file, 
                        X_train=X_train, y_train=y_train, train_lengths=train_lengths,
                        X_val=X_val, y_val=y_val, val_lengths=val_lengths,
                        y_all=y_all)
    
    print(f"\nSaved dataset to {out_file}")
    print(f"Train: {len(X_train)} frames in {len(train_lengths)} segments")
    print(f"Val:   {len(X_val)} frames in {len(val_lengths)} segments")


if __name__ == '__main__':
    if not RECORDINGS_DIR.exists():
        raise SystemExit(f"Recordings directory '{RECORDINGS_DIR}' does not exist.")
    build_dataset(RECORDINGS_DIR, DATASET_DIR, TRAIN_FRAC)
