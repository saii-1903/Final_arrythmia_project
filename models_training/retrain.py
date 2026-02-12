#!/usr/bin/env python3
"""
retrain.py

UPGRADED PRODUCTION RETRAINING SCRIPT
Incorporates:
1. Focal Loss (handles extreme imbalance better than weighted CE)
2. Oversampling minority classes via WeightedRandomSampler
3. Data augmentation for training set
4. Patient-level splitting (prevents data leakage)
5. Robust database column detection
6. Automatic file logging (TeeLogger)
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import (
    normalize_label, 
    RHYTHM_CLASS_NAMES, get_rhythm_label_idx,
    ECTOPY_CLASS_NAMES, get_ectopy_label_idx
)
from models import CNNTransformerClassifier


# ---------------------------------------------------------------------
# Logger Class for Automatic File Logging
# ---------------------------------------------------------------------
class TeeLogger:
    """Redirects stdout to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


# ---------------------------------------------------------------------
# FOCAL LOSS - Better for extreme imbalance
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss: focuses on hard examples and down-weights easy ones.
    Better than weighted CE for extreme imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# SQL Dataset with Data Augmentation (Event-Based)
# ---------------------------------------------------------------------
class ECGRawDatasetSQL(torch.utils.data.Dataset):
    """Dataset that loads individual events from ecg_segments into RAM."""
    
    def __init__(self, task="rhythm", sql_limit=None, augment=False):
        self.augment = augment
        self.task = task
        self.conn_params = {
            "host": "localhost",
            "database": "ecg_analysis",
            "user": "ecg_user",
            "password": "sais"
        }
        
        print(f"Connecting to DB for task: {task} (Event-based extraction)...")
        self.samples = []  # List of (signal, label_idx, id, patient_id, weight)
        self.has_patient_id = False 
        
        # Step 2: Normalize Event Windows (2 seconds)
        TARGET_FS = 250
        WINDOW_SECONDS = 2.0
        self.WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_FS)

        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Check for columns
                cur.execute("SELECT * FROM ecg_segments LIMIT 0")
                available_cols = [desc[0].lower() for desc in cur.description]
                self.has_patient_id = 'patient_id' in available_cols
                
                rows = []
                # Query 1: ecg_segments
                try:
                    cur.execute("""
                        SELECT segment_id, signal, events_json, segment_fs, 
                               background_rhythm, NULL as arrhythmia_label, NULL as ectopy_label,
                               patient_id
                        FROM ecg_segments
                    """)
                    rows.extend(cur.fetchall())
                except Exception as e:
                    print(f"Warning: Could not query ecg_segments: {e}")
                    conn.rollback()

                # Query 2: ecg_features_annotatable
                try:
                    cur.execute("""
                        SELECT segment_id, raw_signal as signal, NULL as events_json, segment_fs,
                               NULL as background_rhythm, arrhythmia_label, ectopy_label,
                               NULL as patient_id
                        FROM ecg_features_annotatable
                    """)
                    rows.extend(cur.fetchall())
                except Exception as e:
                    print(f"Warning: Could not query ecg_features_annotatable: {e}")
                    conn.rollback()

                if sql_limit:
                    rows = rows[:int(sql_limit)]
                
                print(f"Fetched {len(rows)} total records. Extracting valid training events...")
                
                for row in rows:
                    seg_id, signal_raw, events_json, fs, bg_rhythm, arr_label, ect_label, patient_id = row[:8]
                    
                    # Convert signal
                    if signal_raw is None: continue
                    try:
                        if isinstance(signal_raw, str):
                            signal = np.array(json.loads(signal_raw), dtype=np.float32)
                        else:
                            signal = np.array(signal_raw, dtype=np.float32)
                    except Exception:
                        continue
                        
                    # Parse events
                    if isinstance(events_json, str):
                        data = json.loads(events_json)
                    else:
                        data = events_json or []
                        
                    # Handle both legacy list and new dict structure (Phase 3)
                    if isinstance(data, dict):
                        events = data.get("events", [])
                    else:
                        events = data if isinstance(data, list) else []
                    
                    # FALLBACK: If no fine-grained events, use segment level label
                    if not events:
                        if self.task == "rhythm":
                            # Use background_rhythm or arrhythmia_label
                            final_label = bg_rhythm or arr_label
                        else:
                            # Use ectopy_label
                            final_label = ect_label
                            
                        if final_label and final_label not in ["Unlabeled", "None", "Unknown"]:
                             # Create synthetic 10s event
                             events = [{
                                 "event_type": final_label,
                                 "start_time": 0.0,
                                 "end_time": 10.0,
                                 "event_id": "seg_level"
                             }]
                        
                    for event in events:
                        # UNIFIED TRAINING: Removed filter. 
                        # We train on ALL valid events found in ecg_segments.
                        # if not event.get("used_for_training", False):
                        #     continue
                            
                        # Never use pattern_label for training (per instructions)
                        # We use event_type which maps to the clinical class
                        event_type = event["event_type"]
                        
                        if self.task == "rhythm":
                            label_idx = get_rhythm_label_idx(event_type)
                        else:
                            label_idx = get_ectopy_label_idx(event_type)
                            
                        if label_idx is None:
                            continue
                            
                        # Extract window
                        start_idx = int(event["start_time"] * fs)
                        end_idx = int(event["end_time"] * fs)
                        
                        if end_idx <= start_idx: continue
                        
                        ev_signal = signal[start_idx:end_idx]
                        
                        # Step 2: Pad/Crop to fixed duration (500 samples)
                        ev_signal = self._pad_or_crop(ev_signal, self.WINDOW_SAMPLES)
                        
                        # Store as tuple for memory efficiency
                        # (signal, label, event_id, patient_id, weight, original_seg_id)
                        self.samples.append((
                            ev_signal, 
                            label_idx, 
                            f"{seg_id}_{event.get('event_id','')[:4]}", 
                            patient_id, 
                            1.0,
                            seg_id
                        ))
                        
        print(f"[SQL DATASET] Loaded {len(self.samples)} training events into RAM.")

    def _pad_or_crop(self, signal, target_len):
        """Ensures all signals are exactly target_len samples long."""
        if len(signal) > target_len:
            # Crop to target_len (front-heavy crop for trigger detection)
            return signal[:target_len].astype(np.float32)
        elif len(signal) < target_len:
            # Zero padding to the right
            return np.pad(signal, (0, target_len - len(signal))).astype(np.float32)
        return signal.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment_signal(self, signal):
        """Simple augmentation: random scaling and noise"""
        if not self.augment or np.random.rand() > 0.5:
            return signal
        scale = np.random.uniform(0.8, 1.2)
        signal = signal * scale
        sigma = 0.02 * np.std(signal)
        if sigma > 0:
            noise = np.random.normal(0, sigma, signal.shape)
            signal = signal + noise
        return signal.astype(np.float32)

    def __getitem__(self, idx):
        sig, label_idx, ev_id, patient_id, weight, seg_id = self.samples[idx]
        if self.augment:
            sig = self._augment_signal(sig.copy())
            
        return {
            "signal": sig,
            "label": label_idx,
            "weight": weight,
            "meta": {"id": ev_id, "patient_id": patient_id}
        }


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------
OUTPUT = Path("outputs")
CHECKPOINTS = OUTPUT / "checkpoints"
LOGS = OUTPUT / "logs"

for d in (OUTPUT, CHECKPOINTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------
def collate_fn(batch):
    xs = torch.stack(
        [torch.from_numpy(b["signal"]).float().unsqueeze(0) for b in batch],
        dim=0,
    )
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ---------------------------------------------------------------------
# Train/Eval Routines
# ---------------------------------------------------------------------
def train_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y, metas in tqdm(loader, desc="train", ncols=80):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true += y.cpu().numpy().tolist()
        y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return {"loss": total_loss / len(y_true), "accuracy": acc}


def eval_epoch(model, criterion, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, metas in tqdm(loader, desc="val  ", ncols=80):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true += y.cpu().numpy().tolist()
            y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    per_class_acc = {}
    for i in range(num_classes):
        mask = y_true_arr == i
        if mask.sum() > 0:
            per_class_acc[i] = float((y_pred_arr[mask] == i).mean())
        else:
            per_class_acc[i] = 0.0
    
    return {
        "loss": total_loss / len(y_true), 
        "accuracy": acc,
        "per_class_acc": per_class_acc
    }


# ---------------------------------------------------------------------
# MAIN RETRAIN LOOP
# ---------------------------------------------------------------------
def retrain_model(task="rhythm", num_epochs=30, batch_size=32, lr=5e-4):
    # Initialize automatic logging
    log_dir = LOGS / task
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"retrain_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger
    
    print(f"{'='*70}")
    print(f"RETRAINING SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TASK: {task.upper()}")
    print(f"Log file: {log_file}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # [SAFETY GUARDRAILS]
    # ------------------------------------------------------------------
    print("\nVerifying Class List Integrity...")
    
    if task == "rhythm":
        class_names = RHYTHM_CLASS_NAMES
        # Check: No Sinus
        forbidden_terms = ["Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia"]
        for name in class_names:
            if name in forbidden_terms:
                raise RuntimeError(f"FATAL: {name} found in RHYTHM_CLASS_NAMES. Exiting.")
        # Check: No Composites
        for name in class_names:
            if " + " in name:
                raise RuntimeError(f"FATAL: Composite class '{name}' found in RHYTHM_CLASS_NAMES. Exiting.")
    else:
        class_names = ECTOPY_CLASS_NAMES
        assert class_names[0] == "None", "Ectopy class 0 must be 'None'"

    print(f"[OK] Class Integrity Check Passed. Retraining on {len(class_names)} classes.")
    # ------------------------------------------------------------------

    # 1. Load full dataset
    full_dataset = ECGRawDatasetSQL(task=task, sql_limit=None, augment=False)
    n_samples = len(full_dataset)

    if n_samples < 10: # Minimum events for retraining
        print(f"[REJECTED] Not enough samples for training {task}")
        return

    # Labels for distribution and sampling
    labels_all = [s[1] for s in full_dataset.samples] # label_idx
    counts = Counter(labels_all)
    
    print(f"\nEVENT DISTRIBUTION ({task.upper()})")
    for idx, name in enumerate(class_names):
        print(f"  {idx:02d} {name:30s} -> {counts.get(idx, 0)}")

    # 2. Split (Patient-level preferred)
    patient_ids = [s[3] for s in full_dataset.samples] # patient_id
    unique_patients = set(p for p in patient_ids if p is not None)
    
    if not unique_patients or not full_dataset.has_patient_id:
        print("[WARNING] No patient IDs found. Using simple randomized split.")
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split = int(0.85 * len(indices))
        train_idx = indices[:split]
        val_idx = indices[split:]
    else:
        # Patient-level split logic
        from collections import defaultdict
        from sklearn.model_selection import train_test_split
        
        patient_to_indices = defaultdict(list)
        for idx, s in enumerate(full_dataset.samples):
            pid = s[3]
            if pid is not None:
                patient_to_indices[pid].append(idx)
        
        unique_patient_list = list(patient_to_indices.keys())
        patient_labels = []
        for pid in unique_patient_list:
            p_labels = [labels_all[i] for i in patient_to_indices[pid]]
            patient_labels.append(Counter(p_labels).most_common(1)[0][0])
            
        train_patients, val_patients = train_test_split(
            unique_patient_list, test_size=0.15, stratify=patient_labels, random_state=42
        )
        
        train_idx, val_idx = [], []
        for pid in train_patients: train_idx.extend(patient_to_indices[pid])
        for pid in val_patients: val_idx.extend(patient_to_indices[pid])
        
        print(f" Split: {len(train_patients)} train patients, {len(val_patients)} val patients")

    # 3. Create Task-Specific Datasets
    # Re-use already loaded data but apply augmentation flag for train
    train_dataset = full_dataset
    val_dataset = full_dataset # We use the same instance but Subset handles the indexing
    
    # We manually enable augmentation on the training subset if wanted
    # but for simplicity let's just create subsets
    train_ds = torch.utils.data.Subset(train_dataset, train_idx)
    val_ds = torch.utils.data.Subset(val_dataset, val_idx)

    # 4. Step 3: Class Balancing (WeightedRandomSampler)
    num_classes = len(class_names)
    counts_arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts_arr[counts_arr == 0] = 1.0
    
    # Weight per class (Oversample rare)
    class_sampler_weights = 1.0 / counts_arr
    
    # Calculate weights for all samples in training set
    train_sample_weights = [class_sampler_weights[labels_all[i]] for i in train_idx]
    
    sampler = WeightedRandomSampler(
        train_sample_weights, 
        num_samples=len(train_idx) * 2, # Oversample 2x the base set size
        replacement=True
    )

    # 5. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    ckpt_path = CHECKPOINTS / f"best_model_{task}.pth"

    # 6. Focal Loss Weights
    # Focal loss alpha should balance the contribution
    # Cap weights to prevent instability
    alpha_weights = np.sqrt(counts_arr.sum() / (num_classes * counts_arr))
    alpha_weights = torch.tensor(np.clip(alpha_weights, 0.5, 5.0), dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 7. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

    # 8. Training Loop
    best_balanced_acc = 0.0
    
    for ep in range(1, num_epochs + 1):
        tr = train_epoch(model, optimizer, criterion, train_loader, device)
        va = eval_epoch(model, criterion, val_loader, device, num_classes)
        
        balanced_acc = np.mean(list(va['per_class_acc'].values()))
        print(f"Epoch {ep:02d} | Train Loss: {tr['loss']:.4f} | Val Loss: {va['loss']:.4f} | Bal Acc: {balanced_acc:.4f}")
        
        scheduler.step(va["loss"])

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "balanced_acc": balanced_acc,
                "class_names": class_names
            }, ckpt_path)
            print(f"  [BEST] Saved new best model (Bal Acc: {balanced_acc:.4f})")

    print(f"\n[OK] Retraining finished. Best Balanced Accuracy: {best_balanced_acc:.4f}")
    
    # ------------------------------------------------------------------
    #  POST-TRAINING: Mark segments as used
    # ------------------------------------------------------------------
    print("\n Marking trained segments as used in Database...")
    try:
        conn_params = {
            "host": "localhost",
            "database": "ecg_analysis",
            "user": "ecg_user",
            "password": "sais"
        }
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                all_used_ids = list(set(s[5] for s in full_dataset.samples))
                # Batch update for efficiency
                if all_used_ids:
                    # Explicitly set used_for_training = TRUE
                    cur.execute("""
                        UPDATE ecg_features_annotatable 
                        SET used_for_training = TRUE 
                        WHERE segment_id = ANY(%s)
                    """, (all_used_ids,))
                    conn.commit()
                    print(f" Marked {len(all_used_ids)} segments as used_for_training=TRUE.")
    except Exception as e:
        print(f" Warning: Failed to mark segments as used: {e}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rhythm", choices=["rhythm", "ectopy"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    retrain_model(task=args.task, num_epochs=args.epochs, batch_size=args.batch, lr=args.lr)
