import numpy as np
import joblib
import os
import re

BASE = '/Users/devkaransinghsarkaria/kitchen_safety'
clf = joblib.load(os.path.join(BASE, 'wokwi_sim', 'kitchen_clf.pkl'))

CLASS_NAMES = ["Normal Cooking", "Milk Boilover",
               "Gas Leak", "Timeout Risk", "Flame-out"]

W = 30
CONF_THRESH = 59

# ── ROBUST PARSER (handles both new and old formats) ──────────
def parse_line(line):
    try:
        if '|' not in line:
            return None
        if any(x in line for x in ['Time', '===', '---', '!!', 'Reason', 'Smart', 'Layer', 'Scenario']):
            return None

        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            return None

        # Clean numeric fields: remove 's', 'C', 'ppm', trailing spaces
        time_str = re.sub(r'[^0-9.]', '', parts[0])  # remove everything except digits and dot
        temp_str = re.sub(r'[^0-9.\-]', '', parts[1])
        gas_str  = re.sub(r'[^0-9.\-]', '', parts[2])

        time_s = float(time_str)
        temp   = float(temp_str)
        gas    = float(gas_str)

        presence = 1.0 if 'YES' in parts[3].upper() else 0.0

        # Optional fields: scenario, phase, status
        scenario = "unknown"
        phase    = "unknown"
        status   = "SAFE"

        if len(parts) >= 7:
            scenario = parts[4].strip()
            phase    = parts[5].strip()
            status   = parts[6].strip()
        elif len(parts) == 5:
            # Old format: time|temp|gas|presence|status
            status = parts[4].strip()
        elif len(parts) == 4:
            # Minimal: time|temp|gas|presence
            pass

        return (time_s, temp, gas, presence, scenario, phase, status)

    except Exception:
        return None

# ── FEATURE EXTRACTION — identical to train.py ───────────────
def extract_features(temp_hist, gas_hist, pres_hist, time_s):
    w_temp = np.array(temp_hist[-W:])
    w_gas  = np.array(gas_hist[-W:])
    w_pres = np.array(pres_hist[-W:])
    n = len(w_temp)
    return np.array([[
        np.mean(w_temp),
        np.mean(w_gas),
        np.polyfit(np.arange(n), w_temp, 1)[0],
        np.polyfit(np.arange(n), w_gas, 1)[0],
        np.max(w_temp),
        np.mean(w_pres),
        np.std(w_temp),
        np.std(w_gas),
        min(time_s / 1200.0, 1.0)
    ]])

# ── READ serial_output.txt ────────────────────────────────────
with open(os.path.join(BASE, 'wokwi_sim', 'serial_output.txt'), 'r') as f:
    lines = f.readlines()

temp_hist, gas_hist, pres_hist = [], [], []

rows_log = []
ml_fired_at = None
l1_fired_at = None
ml_fired_label = None
ml_fired_conf  = None

print("=" * 75)
print("  LAYER 1 + LAYER 2 INTEGRATION — WOKWI SERIAL OUTPUT")
print("  Source: serial_output.txt")
print("=" * 75)
print(
    f"{'Time':>5} | "
    f"{'Scenario':<12} | "
    f"{'Phase':<10} | "
    f"{'Temp':>6} | "
    f"{'Gas':>7} | "
    f"{'PIR':>3} | "
    f"{'Layer 1':<20} | "
    f"{'Layer 2 ML':<16} | "
    f"{'Conf':>5} | "
    f"{'Match'}"
)
print("-" * 75)

for line in lines:
    parsed = parse_line(line.strip())
    if not parsed:
        continue

    time_s, temp, gas, presence, scenario_name, phase, l1_status = parsed
    temp_hist.append(temp)
    gas_hist.append(gas)
    pres_hist.append(presence)

    if len(temp_hist) < 5:
        print(
            f"{int(time_s):>4}s | "
            f"{scenario_name:<12} | "
            f"{phase:<10} | "
            f"{temp:>5.1f}C | "
            f"{gas:>6.0f}ppm | "
            f"{'YES' if presence else 'NO':>3} | "
            f"{l1_status:<20} | "
            f"{'Collecting...':<16}"
        )
        continue

    features = extract_features(temp_hist, gas_hist, pres_hist, time_s)
    probs = clf.predict_proba(features)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx] * 100)

    if confidence < CONF_THRESH:
        ml_label = "UNCERTAIN"
    else:
        ml_label = CLASS_NAMES[pred_idx]

    l1_danger = any(x in l1_status for x in ['FIRED', 'LEAK', 'SHUTOFF', 'TIMEOUT', 'FLAME'])
    l2_danger = (ml_label != "UNCERTAIN") and (pred_idx in [1, 2, 3, 4])

    # ── NEW LOGIC: only record ML danger if it matches the actual scenario ──
    # Map scenario_name (from log) to the corresponding class name
    scenario_to_class = {
        "GAS_LEAK": "Gas Leak",
        "BOILOVER": "Milk Boilover",
        "TIMEOUT": "Timeout Risk",
        "FLAME_OUT": "Flame-out",
        "NORMAL": "Normal Cooking"
    }
    expected_class = scenario_to_class.get(scenario_name.upper(), None)

    # Only fire if the predicted label matches the expected scenario
    if l2_danger and expected_class is not None and ml_label == expected_class:
        if ml_fired_at is None:
            ml_fired_at = int(time_s)
            ml_fired_label = ml_label
            ml_fired_conf = confidence

    if l1_danger and l1_fired_at is None:
        l1_fired_at = int(time_s)

    match = "✓" if (l1_danger == l2_danger) else "✗"

    rows_log.append({
        'time': int(time_s),
        'temp': temp,
        'gas': gas,
        'presence': presence,
        'scenario': scenario_name,
        'phase': phase,
        'l1_status': l1_status,
        'ml_label': ml_label,
        'confidence': confidence,
        'l1_danger': l1_danger,
        'l2_danger': l2_danger,
        'match': match
    })

    print(
        f"{int(time_s):>4}s | "
        f"{scenario_name:<12} | "
        f"{phase:<10} | "
        f"{temp:>5.1f}C | "
        f"{gas:>6.0f}ppm | "
        f"{'YES' if presence else 'NO':>3} | "
        f"{l1_status:<20} | "
        f"{ml_label:<16} | "
        f"{confidence:>4.0f}% | "
        f"{match}"
    )

# ── SUMMARY ────────────────────────────────────────────────────
print("-" * 75)
print(f"\n  INTEGRATION RESULTS")
print(f"  {'─'*40}")

if ml_fired_at is not None and l1_fired_at is not None:
    lead = l1_fired_at - ml_fired_at
    print(f"  Layer 2 ML first flagged danger (correct class) : t = {ml_fired_at}s  "
          f"({ml_fired_label}, {ml_fired_conf:.0f}% confidence)")
    print(f"  Layer 1 Rule engine fired          : t = {l1_fired_at}s  (threshold crossed)")
    if lead > 0:
        print(f"  ⏱  ML gave {lead}s early warning before Layer 1 threshold breach")
    elif lead == 0:
        print(f"  Both layers detected danger simultaneously")
    else:
        print(f"  Rule engine fired {abs(lead)}s before ML flagged danger")
elif l1_fired_at is not None:
    print(f"  Layer 1 fired at t={l1_fired_at}s — ML never crossed the confidence "
          f"threshold for the correct class in this run.")
elif ml_fired_at is not None:
    print(f"  Layer 2 ML flagged danger at t={ml_fired_at}s — Layer 1 never fired "
          f"in this run.")
else:
    print(f"  Neither layer detected danger in this run.")

print(f"\n  Agreement breakdown:")
if rows_log:
    seg_start = rows_log[0]['time']
    seg_state = (rows_log[0]['l1_danger'], rows_log[0]['ml_label'])
    for i in range(1, len(rows_log) + 1):
        changed = (i == len(rows_log)) or \
                  ((rows_log[i]['l1_danger'], rows_log[i]['ml_label']) != seg_state)
        if changed:
            seg_end = rows_log[i-1]['time']
            l1_word = "danger" if seg_state[0] else "SAFE"
            label_range = f"t={seg_start}-{seg_end}s" if seg_end != seg_start else f"t={seg_start}s"
            print(f"  {label_range:<14}: Layer 1 = {l1_word:<6} | "
                  f"Layer 2 ML = '{seg_state[1]}'")
            if i < len(rows_log):
                seg_start = rows_log[i]['time']
                seg_state = (rows_log[i]['l1_danger'], rows_log[i]['ml_label'])

agree_count = sum(1 for r in rows_log if r['match'] == '✓')
total_count = len(rows_log)
agree_pct = (agree_count / total_count * 100) if total_count else 0

print(f"\n  Overall row-level agreement     : {agree_count}/{total_count} ({agree_pct:.0f}%)")

if l1_fired_at is not None:
    conf_at_trigger = next((r['confidence'] for r in rows_log if r['time'] == l1_fired_at), None)
    label_at_trigger = next((r['ml_label'] for r in rows_log if r['time'] == l1_fired_at), None)
    if conf_at_trigger is not None:
        print(f"  ML state at Layer 1 trigger      : '{label_at_trigger}' at {conf_at_trigger:.0f}% confidence")

if ml_fired_at is not None and l1_fired_at is not None and l1_fired_at > ml_fired_at:
    lead = l1_fired_at - ml_fired_at
    print(f"\n  ← For your paper: ML detected '{ml_fired_label}' pattern {lead}s")
    print(f"     before the deterministic rule engine threshold was breached")
    print(f"     (ML confidence at first detection: {ml_fired_conf:.0f}%).")

print("=" * 75)