import pandas as pd
import numpy as np
import os

# ── LABEL MAP — assign ground truth label to each serial log ──
# Label matches Part 2 class indices exactly:
# 0=Normal Cooking, 1=Milk Boilover, 2=Gas Leak, 3=Timeout Risk, 4=Flame-out

LOG_FILES = {
    'serial_normal.txt':   0,   # Normal Cooking
    'serial_output.txt':   2,   # Gas Leak
    'serial_flameout.txt': 4,   # Flame-out
    'serial_timeout.txt':  3,   # Timeout Risk
}

CLASS_NAMES = {0: "Normal Cooking", 1: "Milk Boilover",
               2: "Gas Leak",       3: "Timeout Risk", 4: "Flame-out"}

W = 30  # same window as Part 2

def parse_line(line):
    try:
        if '|' not in line:
            return None
        if any(x in line for x in ['Time','===','---','!!','Reason','Smart','Layer','Scenario']):
            return None
        parts = line.split('|')
        if len(parts) < 4:
            return None
        time_s   = float(parts[0].strip().replace('s','').replace('\t',''))
        temp     = float(parts[1].strip().replace('C','').replace('\t',''))
        gas      = float(parts[2].strip().replace('ppm','').replace('\t',''))
        presence = 1.0 if 'YES' in parts[3].upper() else 0.0
        return time_s, temp, gas, presence
    except:
        return None

def extract_features_from_log(filepath, label):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    temp_hist, gas_hist, pres_hist, time_hist = [], [], [], []

    for line in lines:
        parsed = parse_line(line.strip())
        if not parsed:
            continue
        time_s, temp, gas, presence = parsed
        temp_hist.append(temp)
        gas_hist.append(gas)
        pres_hist.append(presence)
        time_hist.append(time_s)

    if len(temp_hist) < W:
        print(f"  ⚠ {filepath}: only {len(temp_hist)} points, need {W}. Skipping.")
        return pd.DataFrame()

    rows = []
    total_t = time_hist[-1] if time_hist[-1] > 0 else 1

    for i in range(W, len(temp_hist)):
        w_temp = np.array(temp_hist[i-W:i])
        w_gas  = np.array(gas_hist[i-W:i])
        w_pres = np.array(pres_hist[i-W:i])

        mean_temp = np.mean(w_temp)
        mean_gas  = np.mean(w_gas)
        dT_dt     = np.polyfit(np.arange(W), w_temp, 1)[0]
        dG_dt     = np.polyfit(np.arange(W), w_gas,  1)[0]
        max_temp  = np.max(w_temp)
        pres_val  = np.mean(w_pres)
        time_norm = min(time_hist[i] / total_t, 1.0)

        rows.append({
            'mean_temp': mean_temp,
            'mean_gas':  mean_gas,
            'dT_dt':     dT_dt,
            'dG_dt':     dG_dt,
            'max_temp':  max_temp,
            'presence':  pres_val,
            'time_norm': time_norm,
            'label':     label,
            'class_name': CLASS_NAMES[label],
            'source':    os.path.basename(filepath)
        })

    return pd.DataFrame(rows)

# ── PROCESS ALL LOGS ──────────────────────────────────────────
all_dfs = []
print("Processing Wokwi serial logs...")
print("─" * 45)

for filename, label in LOG_FILES.items():
    filepath = f'/Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim/{filename}'
    if not os.path.exists(filepath):
        print(f"  ✗ NOT FOUND: {filename}")
        continue
    df = extract_features_from_log(filepath, label)
    if not df.empty:
        print(f"  ✓ {filename:<25} → {len(df):>4} samples — {CLASS_NAMES[label]}")
        all_dfs.append(df)

if not all_dfs:
    print("ERROR: No data extracted. Check file paths.")
else:
    wokwi_df = pd.concat(all_dfs, ignore_index=True)
    out_path = '/Users/devkaransinghsarkaria/kitchen_safety/wokwi_sim/wokwi_logs.csv'
    wokwi_df.to_csv(out_path, index=False)
    print("─" * 45)
    print(f"\n✅ Exported {len(wokwi_df)} samples to wokwi_logs.csv")
    print(f"\n  Class distribution:")
    print(wokwi_df['class_name'].value_counts().to_string())
