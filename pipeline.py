# ============================================================
# pipeline.py — Smart Kitchen Safety System
# Run this after EVERY new Wokwi simulation.
# Auto-detects new serial_*.txt files, adds to dataset,
# retrains model, saves new kitchen_clf.pkl, runs bridge.py
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import subprocess
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings('ignore')

BASE      = '/Users/devkaransinghsarkaria/kitchen_safety'
WOKWI_DIR = f'{BASE}/wokwi_sim'
DATA_DIR  = f'{BASE}/data'
MODEL     = f'{BASE}/wokwi_sim/kitchen_clf.pkl'
BRIDGE    = f'{BASE}/bridge.py'

SYNTHETIC = f'{DATA_DIR}/synthetic.csv'
WOKWI_CSV = f'{WOKWI_DIR}/wokwi_logs.csv'
COMBINED  = f'{DATA_DIR}/combined_dataset.csv'

W = 30

CLASS_NAMES = {
    0:"Normal Cooking", 1:"Milk Boilover",
    2:"Gas Leak",       3:"Timeout Risk", 4:"Flame-out"
}

LABEL_MAP = {
    'serial_normal':        0,
    'serial_boilover':      1,
    'serial_output':        2,
    'serial_gasleak':       2,
    'serial_slowgas':       2,

    'serial_nearboil':      0,
    'serial_hotbutsafe':    0,
    'serial_falsealarm':    0,
    'serial_sensornoise':   0,
    'serial_mixed1': 0,
    'serial_chaotic': 2,
    'serial_timeout':       3,
    'serial_returnuser':    3,

    'serial_flameout':      4,
    
    'serial_partialflameout': 4,
}
FEATURE_COLS = ['mean_temp','mean_gas','dT_dt','dG_dt',
                'max_temp','presence','time_norm']

def parse_line(line):
    try:
        if '|' not in line: return None
        if any(x in line for x in ['Time','===','---','!!',
               'Reason','Smart','Layer','Scenario']): return None
        parts = line.split('|')
        if len(parts) < 4: return None
        time_s   = float(parts[0].strip().replace('s','').replace('\t',''))
        temp     = float(parts[1].strip().replace('C','').replace('\t',''))
        gas      = float(parts[2].strip().replace('ppm','').replace('\t',''))
        presence = 1.0 if 'YES' in parts[3].upper() else 0.0
        return time_s, temp, gas, presence
    except: return None

def extract_from_log(filepath, label):
    with open(filepath,'r') as f: lines = f.readlines()
    th,gh,ph,tih = [],[],[],[]
    for line in lines:
        p = parse_line(line.strip())
        if p: tih.append(p[0]); th.append(p[1]); gh.append(p[2]); ph.append(p[3])
    if len(th) <= W: return pd.DataFrame()
    rows = []
    total_t = tih[-1] if tih[-1] > 0 else 1
    for i in range(W, len(th)):
        wt=np.array(th[i-W:i]); wg=np.array(gh[i-W:i]); wp=np.array(ph[i-W:i])
        rows.append({
            'mean_temp': np.mean(wt), 'mean_gas': np.mean(wg),
            'dT_dt':     np.polyfit(np.arange(W),wt,1)[0],
            'dG_dt':     np.polyfit(np.arange(W),wg,1)[0],
            'max_temp':  np.max(wt),  'presence': np.mean(wp),
            'time_norm': min(tih[i]/total_t, 1.0),
            'label': label, 'class_name': CLASS_NAMES[label],
            'source': os.path.basename(filepath)
        })
    return pd.DataFrame(rows)

# ── STEP 1: Scan for new serial logs ─────────────────────────
print("="*55)
print("  PIPELINE — Smart Kitchen Safety System")
print("="*55)
print("\n[1/5] Scanning for new Wokwi serial logs...")

existing = pd.read_csv(WOKWI_CSV) if os.path.exists(WOKWI_CSV) else pd.DataFrame()
already  = set(existing['source'].unique()) if not existing.empty else set()
print(f"  Existing database: {len(existing)} samples from {len(already)} files")

serial_files = sorted([f for f in os.listdir(WOKWI_DIR)
                        if f.startswith('serial_') and f.endswith('.txt')])
new_dfs = []

for fname in serial_files:
    if fname in already:
        print(f"  ↩ Already processed: {fname}")
        continue
    label = next((lbl for key,lbl in LABEL_MAP.items() if key in fname), None)
    if label is None:
        print(f"  ⚠ Unknown: {fname} — add to LABEL_MAP to include")
        continue
    df = extract_from_log(os.path.join(WOKWI_DIR, fname), label)
    if df.empty:
        print(f"  ✗ {fname:<30} — too few points (<{W})")
    else:
        print(f"  + {fname:<30} → {len(df):>4} samples — {CLASS_NAMES[label]}")
        new_dfs.append(df)

# ── STEP 2: Update wokwi_logs.csv ────────────────────────────
print(f"\n[2/5] Updating wokwi_logs.csv...")
if new_dfs:
    new_data = pd.concat(new_dfs, ignore_index=True)
    updated  = pd.concat([existing, new_data], ignore_index=True) \
               if not existing.empty else new_data
    updated.to_csv(WOKWI_CSV, index=False)
    print(f"  Added {len(new_data)} new samples")
    print(f"  Total Wokwi database: {len(updated)} samples")
else:
    updated = existing
    print(f"  No new files — database unchanged ({len(updated)} samples)")

# ── STEP 3: Merge with synthetic ─────────────────────────────
print(f"\n[3/5] Merging datasets...")
if not os.path.exists(SYNTHETIC):
    print("  ✗ synthetic.csv not found — run train.py first!")
    exit()

syn_df   = pd.read_csv(SYNTHETIC)
combined = pd.concat([syn_df, updated], ignore_index=True) \
           if not updated.empty else syn_df
combined.to_csv(COMBINED, index=False)
print(f"  Synthetic : {len(syn_df):>7} samples")
print(f"  Wokwi     : {len(updated):>7} samples")
print(f"  Combined  : {len(combined):>7} samples")

# ── STEP 4: Retrain ──────────────────────────────────────────
print(f"\n[4/5] Retraining Random Forest...")
X = combined[FEATURE_COLS].values
y = combined['label'].values

X_tr,X_te,y_tr,y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
clf.fit(X_tr, y_tr)

train_acc = clf.score(X_tr, y_tr)*100
test_acc  = clf.score(X_te, y_te)*100
cv        = cross_val_score(clf, X, y, cv=5)*100

print(f"  Training accuracy   : {train_acc:.2f}%")
print(f"  Test accuracy       : {test_acc:.2f}%")
print(f"  CV accuracy (5-fold): {cv.mean():.2f}% ± {cv.std():.2f}%")

joblib.dump(clf, MODEL)
print(f"  ✅ Model saved: wokwi_sim/kitchen_clf.pkl")

# ── STEP 5: Run bridge.py ─────────────────────────────────────
print(f"\n[5/5] Running bridge.py...")
print("─"*55)
subprocess.run(['python3', BRIDGE])
print("─"*55)

print(f"\n  PIPELINE COMPLETE")
print(f"  Dataset: {len(combined)} samples "
      f"({len(syn_df)} synthetic + {len(updated)} Wokwi)")
print(f"  Accuracy: {test_acc:.2f}%")
print(f"\n  NEXT TIME: run a new Wokwi scenario, then:")
print(f"  python3 pipeline.py")
print("="*55)
