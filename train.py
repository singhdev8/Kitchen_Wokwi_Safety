# ============================================================
# train.py — Smart Kitchen Safety System
# Runs Part 1 (simulation) + Part 2 (ML) + Part 3 (prediction)
# Run once to generate all figures + kitchen_clf.pkl
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── PATHS ─────────────────────────────────────────────────────
BASE     = '/Users/devkaransinghsarkaria/kitchen_safety'
DATA_DIR = f'{BASE}/data'
OUT_DIR  = f'{BASE}/outputs'
MODEL    = f'{BASE}/wokwi_sim/kitchen_clf.pkl'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

dt = 1
W  = 30

CLASS_NAMES = ["Normal Cooking", "Milk Boilover",
               "Gas Leak", "Timeout Risk", "Flame-out"]

FEATURE_COLS = ['mean_temp','mean_gas','dT_dt','dG_dt',
                'max_temp','presence','time_norm']

# ════════════════════════════════════════════════════════════════
# PART 1 — LAYER 1 SIMULATION
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 1 — Layer 1 Rule Engine Simulation")
print("="*60)

# ── Scenario data ─────────────────────────────────────────────
t1 = np.arange(0, 1200, dt)
temp1 = 82 * (1 - np.exp(-t1 / 400)) + 25
gas1  = 300 + 20 * np.sin(t1 / 60)
presence1 = np.ones(len(t1))

t2 = np.arange(0, 600, dt)
temp2 = 30 + 75 * (1 - np.exp(-t2 / 40))
gas2  = 310 + 10 * np.sin(t2 / 30)
presence2 = np.ones(len(t2))

t3 = np.arange(0, 900, dt)
temp3 = 30 + 2 * np.random.randn(len(t3))
gas3  = 200 + 0.3 * t3
presence3 = np.zeros(len(t3))

t4 = np.arange(0, 4000, dt)
temp4 = 50 * (1 - np.exp(-t4 / 300)) + 25
gas4  = 310 + 15 * np.sin(t4 / 60)
presence4 = np.where(t4 < 600, 1, 0)

t5 = np.arange(0, 600, dt)

temp5 = np.where(
    t5 < 160,
    30 + 40 * (1 - np.exp(-t5 / 120)),
    70 * np.exp(-(t5 - 160) / 40) + 20
)
temp5 += np.random.normal(0, 0.8, len(t5))

gas5 = np.where(
    t5 < 220,
    300 + np.random.normal(0, 3, len(t5)),
    300 + 2.2 * (t5 - 220) + np.random.normal(0, 5, len(t5))
)

presence5 = np.ones(len(t5))

print("✅ All 5 scenarios generated!")

# ── Rule engine ───────────────────────────────────────────────
TEMP_BOILOVER  = 88
TEMP_RISE_RATE = 0.25
GAS_LEAK_THRESH= 400
TIMEOUT_LIMIT  = 2700
TEMP_DROP_RATE = -0.5
RULE_WINDOW    = 5

def run_rule_engine(t, temp, gas, presence, scenario_name):
    events = []
    state  = np.zeros(len(t))
    no_presence_timer = 0
    triggered = False

    for i in range(1, len(t)):
        if triggered:
            state[i] = 1
            continue
        w_start = max(0, i - RULE_WINDOW)
        dT_avg  = (temp[i] - temp[w_start]) / ((i - w_start) * dt)
        dG      = (gas[i]  - gas[i-1]) / dt

        if dT_avg < TEMP_DROP_RATE and dG > 0.5 and gas[i] > 300:
            events.append((t[i], "FLAME-OUT — Emergency Shutoff"))
            state[i] = 1
            triggered = True
            continue
        if temp[i] >= TEMP_BOILOVER and dT_avg >= TEMP_RISE_RATE and gas[i] > 200:
            events.append((t[i], "OVERFLOW RISK — Shutoff"))
            state[i] = 1
            triggered = True
            continue
        if gas[i] >= GAS_LEAK_THRESH and temp[i] < 40:
            events.append((t[i], "GAS LEAK (no flame) — Shutoff"))
            state[i] = 1
            triggered = True
            continue
        if presence[i] == 0 and gas[i] > 200:
            no_presence_timer += dt
        else:
            no_presence_timer = 0

        if no_presence_timer >= TIMEOUT_LIMIT:
            events.append((t[i], "TIMEOUT 45min — Shutoff"))
            state[i] = 1
            triggered = True
            continue   

    if not events:
        events.append((t[-1], "NO TRIGGER — System Safe ✓"))

    print(f"  {scenario_name:<22} → {events[0][1]} at t={events[0][0]:.0f}s")
    return state, events

print("\nRunning rule engine...")
print("─" * 60)
state1, ev1 = run_rule_engine(t1, temp1, gas1, presence1, "Normal Cooking")
state2, ev2 = run_rule_engine(t2, temp2, gas2, presence2, "Milk Boilover")
state3, ev3 = run_rule_engine(t3, temp3, gas3, presence3, "Gas Leak")
state4, ev4 = run_rule_engine(t4, temp4, gas4, presence4, "Timeout")
state5, ev5 = run_rule_engine(t5, temp5, gas5, presence5, "Flame-out")
print("─" * 60)

# ── Figure 1: Simulation dashboard ───────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 22))
fig.suptitle("Smart Kitchen Safety System — Simulation Results",
             fontsize=16, fontweight='bold', y=0.98)

scenarios_plot = [
    (t1,temp1,gas1,state1,ev1,"Scenario 1: Normal Cooking"),
    (t2,temp2,gas2,state2,ev2,"Scenario 2: Milk Boilover"),
    (t3,temp3,gas3,state3,ev3,"Scenario 3: Gas Leak / No Flame"),
    (t4,temp4,gas4,state4,ev4,"Scenario 4: Timeout (Person Leaves)"),
    (t5,temp5,gas5,state5,ev5,"Scenario 5: Flame-out with Gas Flowing"),
]

for ax, (t,temp,gas,state,events,title) in zip(axes, scenarios_plot):
    ax2 = ax.twinx()
    ax.plot(t, temp,        'r-',  linewidth=1.5, label='Temperature (°C)')
    ax2.plot(t, gas,        'b--', linewidth=1.5, label='Gas (ppm)', alpha=0.8)
    ax.plot(t, state * 100, 'g-',  linewidth=2.5, label='System State (×100)')
    for (trigger_t, label) in events:
        if "Safe" not in label:
            ax.axvline(x=trigger_t, color='orange', linestyle='--', linewidth=2)
            ax.text(trigger_t + max(t)*0.01, 15, label,
                    fontsize=7, color='darkorange', rotation=90, va='bottom')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Temperature (°C) / State", color='red')
    ax2.set_ylabel("Gas Concentration (ppm)", color='blue')
    ax.set_ylim(0, 120)
    ax2.set_ylim(0, max(gas) * 1.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f'{OUT_DIR}/kitchen_safety_simulation.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {out}")

# ════════════════════════════════════════════════════════════════
# PART 2 — LAYER 2 ML CLASSIFIER
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 2 — Layer 2 ML Classifier")
print("="*60)

# ── Generators ────────────────────────────────────────────────
def gen_normal_cooking():

    ambient = np.random.uniform(20, 30)

    tau = np.random.uniform(250, 550)

    rise = np.random.uniform(45, 70)

    noise = np.random.uniform(1.0, 4.0)

    gas_base = np.random.uniform(270, 330)

    gas_amp = np.random.uniform(5, 30)

    t = np.arange(0, np.random.randint(800, 1500), dt)

    temp = rise*(1-np.exp(-t/tau)) + ambient

    temp += np.random.randn(len(t))*noise

    gas = gas_base + gas_amp*np.sin(t/np.random.uniform(40,90))

    gas += np.random.randn(len(t))*8
    gas += np.linspace(
    np.random.uniform(-20,20),
    np.random.uniform(-20,20),
    len(t)
    )

    presence = np.random.choice(
        [0,1],
        size=len(t),
        p=[0.15,0.85]
    )

    return t,temp,gas,presence

def gen_milk_boilover():
    ambient  = 25 + np.random.uniform(-3, 3)
    tau      = np.random.uniform(30, 55)
    rise     = np.random.uniform(65, 85)
    noise    = np.random.uniform(1.0, 4.0)
    gas_base = np.random.uniform(300, 320)
    gas_amp  = np.random.uniform(5, 15)
    t = np.arange(0, np.random.randint(400, 700), dt)
    temp = rise*(1-np.exp(-t/tau)) + ambient + np.random.randn(len(t))*noise
    gas  = gas_base + gas_amp*np.sin(t/np.random.uniform(20, 40))
    return t, temp, gas, np.ones(len(t))

def gen_gas_leak():

    ambient = np.random.uniform(22,60)

    noise = np.random.uniform(1.0,4.0)

    gas_start = np.random.uniform(100,250)

    gas_rate = np.random.uniform(0.1,0.5)

    t = np.arange(0,np.random.randint(600,1200),dt)

    temp = ambient

    temp += 10*np.sin(t/120)

    temp += np.random.randn(len(t))*noise

    gas = gas_start + gas_rate*t

    gas += np.random.randn(len(t))*10

    presence = np.random.choice(
    [0,1],
    size=len(t),
    p=[0.45,0.55]
    )

    return t,temp,gas,presence

def gen_timeout():
    ambient  = 24 + np.random.uniform(-3, 3)
    tau      = np.random.uniform(250, 400)
    rise     = np.random.uniform(40, 60)
    noise = np.random.uniform(1.0, 4.0)
    gas_base = np.random.uniform(300, 320)
    gas_amp  = np.random.uniform(8, 20)
    leave_t  = np.random.randint(400, 800)
    t = np.arange(0, np.random.randint(1200, 2200), dt)
    temp = rise*(1-np.exp(-t/tau)) + ambient + np.random.randn(len(t))*noise
    gas  = gas_base + gas_amp*np.sin(t/np.random.uniform(50, 70))
    return t, temp, gas, np.where(t < leave_t, 1, 0)

def gen_flame_out():
    ambient  = 25 + np.random.uniform(-3, 3)
    rise     = np.random.uniform(50, 70)
    rise_tau = np.random.uniform(120, 180)
    drop_tau = np.random.uniform(70, 130)
    noise = np.random.uniform(1.0, 4.0)
    gas_base = np.random.uniform(290, 310)
    gas_rate = np.random.uniform(1.0, 2.0)
    switch_t = np.random.randint(250, 350)
    total_t  = np.random.randint(500, 700)
    t = np.arange(0, total_t, dt)
    temp = np.where(t < switch_t,
                    ambient + rise*(1-np.exp(-t/rise_tau)),
                    (ambient+rise)*np.exp(-(t-switch_t)/drop_tau) + ambient - 3)
    temp += np.random.randn(len(t))*noise
    gas  = np.where(t < switch_t, gas_base, gas_base + gas_rate*(t-switch_t))
    return t, temp, gas, np.ones(len(t))

GENERATORS = [
    (gen_normal_cooking, 0, "Normal Cooking"),
    (gen_milk_boilover,  1, "Milk Boilover"),
    (gen_gas_leak,       2, "Gas Leak"),
    (gen_timeout,        3, "Timeout Risk"),
    (gen_flame_out,      4, "Flame-out"),
]

# ── Feature extraction ────────────────────────────────────────
def extract_features(t, temp, gas, presence, W):
    X = []
    for i in range(W, len(t)):
        wt = temp[i-W:i]; wg = gas[i-W:i]; wp = presence[i-W:i]
        X.append([
            np.mean(wt), np.mean(wg),
            np.polyfit(np.arange(W), wt, 1)[0],
            np.polyfit(np.arange(W), wg, 1)[0],
            np.max(wt), np.mean(wp), t[i]/t[-1]
        ])
    return np.array(X)

# ── Build dataset ─────────────────────────────────────────────
RUNS = 4
X_all, y_all = [], []
print("Generating synthetic dataset...")

for (gen_fn, label, name) in GENERATORS:
    total = 0
    for _ in range(RUNS):
        t, temp, gas, presence = gen_fn()
        Xsc = extract_features(t, temp, gas, presence, W)
        X_all.append(Xsc)
        y_all.append(np.full(len(Xsc), label))
        total += len(Xsc)
    print(f"  {name:<20} → {RUNS} runs, {total} samples")

X = np.vstack(X_all)
y = np.concatenate(y_all)
print(f"\n✅ Synthetic dataset: {X.shape[0]} samples × {X.shape[1]} features")

# ── Save synthetic.csv ────────────────────────────────────────
syn_df = pd.DataFrame(X, columns=FEATURE_COLS)
syn_df['label']      = y
syn_df['class_name'] = syn_df['label'].map({i:n for i,n in enumerate(CLASS_NAMES)})
syn_df['source']     = 'synthetic'
syn_df.to_csv(f'{DATA_DIR}/synthetic.csv', index=False)
print(f"✅ Saved: data/synthetic.csv")

# ── Check for Wokwi data and merge ───────────────────────────
wokwi_csv = f'{BASE}/wokwi_sim/wokwi_logs.csv'
if os.path.exists(wokwi_csv):
    wok_df   = pd.read_csv(wokwi_csv)
    combined = pd.concat([syn_df, wok_df], ignore_index=True)
    combined.to_csv(f'{DATA_DIR}/combined_dataset.csv', index=False)
    X_train_data = combined[FEATURE_COLS].values
    y_train_data = combined['label'].values
    print(f"✅ Merged with Wokwi data → {len(combined)} total samples")
else:
    X_train_data = X
    y_train_data = y
    print("ℹ  No wokwi_logs.csv found — training on synthetic only")

# ── Train ─────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_data, y_train_data, test_size=0.2, random_state=42,
    stratify=y_train_data)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_tr, y_tr)

train_acc = clf.score(X_tr, y_tr) * 100
test_acc  = clf.score(X_te, y_te) * 100
cv        = cross_val_score(clf, X_train_data, y_train_data, cv=5) * 100

print(f"\n{'─'*45}")
print(f"  Training accuracy   : {train_acc:.2f}%")
print(f"  Test accuracy       : {test_acc:.2f}%")
print(f"  CV accuracy (5-fold): {cv.mean():.2f}% ± {cv.std():.2f}%")
print(f"{'─'*45}")

# ── Confusion matrix ──────────────────────────────────────────
y_pred = clf.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
    ax=ax, colorbar=True, cmap='Blues')
ax.set_title("Figure 2 — Confusion Matrix", fontweight='bold', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: outputs/confusion_matrix.png")

# ── Feature importance ────────────────────────────────────────
FEATURE_NAMES = ["Mean Temp","Mean Gas","dT/dt","dG/dt",
                 "Max Temp","Presence","Time (norm)"]
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#d62728' if i==order[0] else '#1f77b4' for i in range(7)]
bars = ax.bar([FEATURE_NAMES[i] for i in order], importances[order]*100,
              color=[colors[r] for r,_ in enumerate(order)])
ax.set_ylabel("Feature Importance (%)")
ax.set_title("Figure 3 — Feature Importance", fontweight='bold', fontsize=12)
ax.set_ylim(0, max(importances)*110)
for bar, imp in zip(bars, importances[order]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{imp*100:.1f}%", ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: outputs/feature_importance.png")

# ── Save model ────────────────────────────────────────────────
joblib.dump(clf, MODEL)
print(f"✅ Saved: wokwi_sim/kitchen_clf.pkl")

# ════════════════════════════════════════════════════════════════
# PART 3 — LAYER 3 PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PART 3 — Layer 3 Prediction Engine")
print("="*60)

t = np.arange(0, 3000, dt)
temp     = np.zeros(len(t))
gas      = np.zeros(len(t))
presence = np.zeros(len(t))
temp_at_500 = 82*(1-np.exp(-500/400)) + 25

for i, ti in enumerate(t):
    if ti < 500:
        temp[i]     = 82*(1-np.exp(-ti/400)) + 25 + np.random.randn()*0.3
        gas[i]      = 300 + 15*np.sin(ti/60)
        presence[i] = 1
    elif ti < 900:
        elapsed     = ti - 500
        temp[i]     = temp_at_500 + (105-temp_at_500)*(1-np.exp(-elapsed/120)) + np.random.randn()*0.3
        gas[i]      = 310 + 10*np.sin(ti/60)
        presence[i] = 1
    else:
        elapsed     = ti - 900
        temp[i]     = 95*np.exp(-elapsed/800) + 28 + np.random.randn()*0.3
        gas[i]      = 310 + 0.05*elapsed
        presence[i] = 0

PRED_WINDOW=30; HORIZON=60; OVERFLOW_THRESH=100; BURN_THRESH=95
BURN_LIMIT=300; K_FORGET=0.0012

T_pred_60    = np.full(len(t), np.nan)
burn_counter = np.zeros(len(t))
p_forget     = np.zeros(len(t))
no_pres_timer= 0; burn_cnt=0
overflow_pred_t=None; overflow_actual_t=None

for i in range(PRED_WINDOW, len(t)):
    w_t   = np.arange(PRED_WINDOW)
    w_temp= temp[i-PRED_WINDOW:i]
    slope, intercept = np.polyfit(w_t, w_temp, 1)
    T_pred_60[i] = slope*(PRED_WINDOW+HORIZON) + intercept
    if T_pred_60[i] > OVERFLOW_THRESH and overflow_pred_t is None:
        overflow_pred_t = t[i]
    if temp[i] >= OVERFLOW_THRESH and overflow_actual_t is None:
        overflow_actual_t = t[i]
    burn_cnt = burn_cnt + dt if temp[i] > BURN_THRESH else 0
    burn_counter[i] = burn_cnt
    if presence[i]==0 and gas[i]>200 and temp[i]<98:
        no_pres_timer += dt
    else:
        no_pres_timer = 0
    p_forget[i] = 1 - np.exp(-K_FORGET*no_pres_timer)

lead_time = overflow_actual_t - overflow_pred_t \
            if (overflow_pred_t and overflow_actual_t) else None

print(f"  Overflow predicted at : t = {overflow_pred_t}s")
print(f"  Overflow actual at    : t = {overflow_actual_t}s")
print(f"  ⏱ Lead time           : {lead_time}s")

# ── Figure 5: Prediction dashboard ───────────────────────────
fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(16,13), sharex=True,
                                  gridspec_kw={'height_ratios':[3,2,2]})
fig.suptitle("Figure 5 — Layer 3 Prediction Dashboard",
             fontsize=13, fontweight='bold')

ax1.plot(t, temp,      'r-',  linewidth=1.5, label='Actual Temp (°C)')
ax1.plot(t, T_pred_60, '--',  color='darkorange', linewidth=1.5,
         label='Predicted Temp at t+60s')
ax1.axhline(OVERFLOW_THRESH, color='black',  linestyle=':', linewidth=1,
            label='Overflow threshold (100°C)')
ax1.axhline(BURN_THRESH,     color='purple', linestyle=':', linewidth=1,
            label='Burn threshold (95°C)')
for xpos, lbl in [(250,'Normal\nCooking'),(700,'Boilover\nPhase'),(1950,'Person Left\n— Gas On')]:
    ax1.text(xpos, 5, lbl, ha='center', fontsize=8, color='gray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
ax1.axvline(500, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(900, color='gray', linestyle=':', alpha=0.5)
if overflow_pred_t:
    ax1.axvline(overflow_pred_t, color='green', linestyle='--', linewidth=2)
    ax1.text(overflow_pred_t+15, 32,
             f'⚠ Predicted\nt={overflow_pred_t:.0f}s', color='green', fontsize=8)
if overflow_actual_t:
    ax1.axvline(overflow_actual_t, color='red', linestyle='--', linewidth=2)
    ax1.text(overflow_actual_t+15, 52,
             f'✖ Actual\nt={overflow_actual_t:.0f}s', color='red', fontsize=8)
if lead_time:
    mid = (overflow_pred_t+overflow_actual_t)/2
    ax1.annotate('', xy=(overflow_actual_t,118), xytext=(overflow_pred_t,118),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(mid, 120, f'Lead time: {lead_time:.0f}s',
             ha='center', fontsize=9, color='green', fontweight='bold')
ax1.set_ylabel("Temperature (°C)"); ax1.set_ylim(0,130)
ax1.legend(loc='upper right', fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_title("Overflow Prediction via 30s Rolling Linear Extrapolation")

burn_trigger = np.where(burn_counter >= BURN_LIMIT)[0]
ax2.plot(t, burn_counter, color='purple', linewidth=1.5, label='Burn counter (s)')
ax2.axhline(BURN_LIMIT, color='red', linestyle='--', linewidth=1.5,
            label=f'Burn threshold ({BURN_LIMIT}s)')
if len(burn_trigger):
    bt = t[burn_trigger[0]]
    ax2.axvline(bt, color='darkred', linestyle='--', linewidth=2)
    ax2.text(bt+20, BURN_LIMIT*0.45,
             f'⚠ Burn risk\nt={bt:.0f}s', color='darkred', fontsize=8)
ax2.set_ylabel("Seconds above 95°C"); ax2.set_ylim(0, BURN_LIMIT*1.4)
ax2.legend(loc='upper right', fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_title("Burn Risk Counter")

amber_i = np.where(p_forget >= 0.75)[0]
red_i   = np.where(p_forget >= 0.95)[0]
ax3.plot(t, p_forget, 'b-', linewidth=1.5, label='P(forgetfulness)')
ax3.axhline(0.75, color='orange', linestyle='--', linewidth=1.5, label='Amber (0.75)')
ax3.axhline(0.95, color='red',    linestyle='--', linewidth=1.5, label='Red (0.95)')
if len(amber_i):
    ax3.axvline(t[amber_i[0]], color='orange', linestyle=':', linewidth=2)
    ax3.text(t[amber_i[0]]+30, 0.60,
             f'Amber\nt={t[amber_i[0]]:.0f}s', color='darkorange', fontsize=8)
if len(red_i):
    ax3.axvline(t[red_i[0]], color='red', linestyle=':', linewidth=2)
    ax3.text(t[red_i[0]]-250, 0.78,
             f'Red\nt={t[red_i[0]]:.0f}s', color='red', fontsize=8)
ax3.set_xlabel("Time (seconds)"); ax3.set_ylabel("Probability")
ax3.set_ylim(0,1.1); ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3); ax3.set_title("Forgetfulness Probability")

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/layer3_prediction_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: outputs/layer3_prediction_dashboard.png")

# ── Figure 6: Zoom ────────────────────────────────────────────
if overflow_pred_t and overflow_actual_t:
    zs = max(0, int(overflow_pred_t)-120)
    ze = min(len(t), int(overflow_actual_t)+120)
    fig2, ax = plt.subplots(figsize=(10,5))
    ax.plot(t[zs:ze], temp[zs:ze],     'r-',  linewidth=2, label='Actual Temp (°C)')
    ax.plot(t[zs:ze], T_pred_60[zs:ze],'--',  color='darkorange',
            linewidth=2, label='Predicted Temp t+60s')
    ax.axhline(100, color='black', linestyle=':', linewidth=1.5,
               label='100°C threshold')
    ax.axvline(overflow_pred_t,   color='green', linestyle='--', linewidth=2,
               label=f'Prediction fires (t={overflow_pred_t:.0f}s)')
    ax.axvline(overflow_actual_t, color='red',   linestyle='--', linewidth=2,
               label=f'Actual overflow (t={overflow_actual_t:.0f}s)')
    ax.annotate('', xy=(overflow_actual_t,113), xytext=(overflow_pred_t,113),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
    ax.text((overflow_pred_t+overflow_actual_t)/2, 116,
            f'⏱ {lead_time:.0f}s warning lead time',
            ha='center', fontsize=11, color='green', fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title("Figure 6 — Overflow Prediction Lead Time Zoom",
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/layer3_overflow_zoom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: outputs/layer3_overflow_zoom.png")

# ── Table 2 ───────────────────────────────────────────────────
burn_lead  = t[burn_trigger[0]] if len(burn_trigger) else "N/A"
amber_lead = t[amber_i[0]]     if len(amber_i)      else "N/A"
red_lead   = t[red_i[0]]       if len(red_i)        else "N/A"

print("\n" + "="*52)
print("  TABLE 2 — Layer 3 Warning Lead Times")
print("─"*52)
if lead_time:
    print(f"  Milk Overflow : {lead_time:.0f}s warning before breach")
if isinstance(burn_lead, (float, np.floating)):
    print(f"  Burn Risk     : fires at t={burn_lead:.0f}s")
if isinstance(amber_lead, (int,float,np.floating)):
    print(f"  Forget Amber  : fires at t={amber_lead:.0f}s  [P=0.75]")
if isinstance(red_lead, (int,float,np.floating)):
    print(f"  Forget Red    : fires at t={red_lead:.0f}s  [P=0.95]")
print("="*52)

print("\n" + "="*60)
print("  TRAIN.PY COMPLETE")
print(f"  Outputs → {OUT_DIR}/")
print(f"  Model   → {MODEL}")
print("="*60)
