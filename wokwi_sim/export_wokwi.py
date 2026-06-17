import pandas as pd
import numpy as np
import os

# ── LABEL MAP — assign ground truth label to each serial log ──
# Label matches Part 2 class indices exactly:
# 0=Normal Cooking, 1=Milk Boilover, 2=Gas Leak, 3=Timeout Risk, 4=Flame-out

LOG_FILES = {
    'serial_normal.txt':      0,
    'serial_boilover.txt':    1,
    'serial_gasleak.txt':     2,
    'serial_timeout.txt':     3,
    'serial_flameout.txt':    4,
}

CLASS_NAMES = {0: "Normal Cooking", 1: "Milk Boilover",
               2: "Gas Leak",       3: "Timeout Risk", 4: "Flame-out"}

W = 30  # window size


def parse_line(line):
    """
    Parse a single line from a serial log.
    Handles both the new pipe‑delimited format (time|temp|gas|presence|scenario|phase|status)
    and the old format with units (e.g., '1s | 69.8C | 325ppm | YES | SAFE').
    """
    try:
        if '|' not in line:
            return None
        if any(x in line for x in ['Time', '===', '---', '!!', 'Reason', 'Smart', 'Layer', 'Scenario']):
            return None

        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            return None

        # --- Clean numeric fields (remove units and extra spaces) ---
        time_s_str = parts[0].replace('s', '').strip()
        temp_str   = parts[1].replace('C', '').strip()
        gas_str    = parts[2].replace('ppm', '').strip()

        time_s = float(time_s_str)
        temp   = float(temp_str)
        gas    = float(gas_str)

        # --- Presence ---
        presence_str = parts[3].strip().upper()
        presence = 1.0 if 'YES' in presence_str else 0.0

        # --- Optional extra fields (scenario, phase, status) ---
        scenario = None
        phase    = None
        status   = None
        if len(parts) >= 7:
            scenario = parts[4].strip()
            phase    = parts[5].strip()
            status   = parts[6].strip()
        elif len(parts) == 5:
            # Old format: time|temp|gas|presence|status
            status = parts[4].strip()

        return (time_s, temp, gas, presence, scenario, phase, status)

    except Exception:
        return None


def extract_features_from_log(filepath, label):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    temp_hist, gas_hist, pres_hist, time_hist = [], [], [], []
    scenario_hist, phase_hist, status_hist = [], [], []

    for line in lines:
        parsed = parse_line(line.strip())
        if not parsed:
            continue
        time_s, temp, gas, presence, scenario, phase, status = parsed
        temp_hist.append(temp)
        gas_hist.append(gas)
        pres_hist.append(presence)
        time_hist.append(time_s)
        scenario_hist.append(scenario if scenario is not None else CLASS_NAMES[label])
        phase_hist.append(phase if phase is not None else "unknown")
        status_hist.append(status if status is not None else "")

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
        temp_std  = np.std(w_temp)
        gas_std   = np.std(w_gas)
        time_norm = min(time_hist[i] / total_t, 1.0)

        rows.append({
            'mean_temp': mean_temp,
            'mean_gas':  mean_gas,
            'dT_dt':     dT_dt,
            'dG_dt':     dG_dt,
            'max_temp':  max_temp,
            'presence':  pres_val,
            'temp_std':  temp_std,
            'gas_std':   gas_std,
            'time_norm': time_norm,
            'scenario':  scenario_hist[i],
            'phase':     phase_hist[i],
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

    if 'scenario' in wokwi_df.columns:
        print("\n  Scenario distribution:")
        print(wokwi_df['scenario'].value_counts().to_string())

    if 'phase' in wokwi_df.columns:
        print("\n  Phase distribution:")
        print(wokwi_df['phase'].value_counts().to_string())