import numpy as np

# --------------------------------------------------
# Helper function to write consistent 7-field format
# --------------------------------------------------
def write_serial(f, t, temp, gas, presence, scenario, phase, status="SAFE"):
    """Write a single line in the 7-field format: time|temp|gas|presence|scenario|phase|status"""
    pir_str = "YES" if presence else "NO"
    # ✅ FIXED: no 's', 'C', 'ppm', no spaces around '|'
    f.write(f"{t}|{temp:.1f}|{gas:.0f}|{pir_str}|{scenario}|{phase}|{status}\n")

# --------------------------------------------------
# FALSE ALARM (Normal Cooking with fluctuations)
# --------------------------------------------------
with open("wokwi_sim/serial_falsealarm.txt", "w") as f:
    for t in range(1, 201):
        temp = 70 + 15*np.sin(t/30) + np.random.randn()*1.5
        gas  = 320 + 20*np.sin(t/25) + np.random.randn()*5
        phase = "Normal" if t < 150 else "Cooling"
        write_serial(f, t, temp, gas, True, "Normal Cooking", phase)

# --------------------------------------------------
# SENSOR NOISE (Normal Cooking with sensor anomalies)
# --------------------------------------------------
with open("wokwi_sim/serial_sensornoise.txt", "w") as f:
    for t in range(1, 201):
        temp = 65 + np.random.randn()*8
        gas  = 300 + np.random.randn()*30
        
        if np.random.rand() < 0.05:
            temp += np.random.randint(15, 35)
        if np.random.rand() < 0.05:
            gas += np.random.randint(80, 150)
        
        phase = "Steady" if 50 < t < 150 else "Ramping"
        write_serial(f, t, temp, max(gas, 0), True, "Normal Cooking", phase)

# --------------------------------------------------
# RETURN USER (Timeout scenario with person leaving and returning)
# --------------------------------------------------
with open("wokwi_sim/serial_returnuser.txt", "w") as f:
    for t in range(1, 201):
        temp = 55 + 10*np.sin(t/40) + np.random.randn()*2
        gas  = 310 + 10*np.sin(t/30)
        
        if t < 60:
            presence = True
            phase = "Cooking"
        elif t < 140:
            presence = False
            phase = "Unattended"
        else:
            presence = True
            phase = "Returned"
        
        write_serial(f, t, temp, gas, presence, "Timeout Risk", phase)

# --------------------------------------------------
# PARTIAL FLAMEOUT (Flame-out with gas continuing)
# --------------------------------------------------
with open("wokwi_sim/serial_partialflameout.txt", "w") as f:
    for t in range(1, 201):
        if t < 70:
            temp = 80 - 0.05*t
            gas = 300
            phase = "Normal"
        elif t < 120:
            temp = 76 - 0.55*(t-70)
            gas = 300 + 2*(t-70)
            phase = "Flame Decreasing"
        elif t < 150:
            temp = 48 + 0.4*(t-120)
            gas = 390
            phase = "Flame Out"
        else:
            temp = 60 - 0.45*(t-150)
            gas = 390 + 1.5*(t-150)
            phase = "Gas Accumulating"
        
        temp += np.random.randn()*1.0
        gas  += np.random.randn()*5
        write_serial(f, t, temp, gas, True, "Flame-out", phase)

# --------------------------------------------------
# FLAMEOUT (Complete flame failure with gas leak)
# --------------------------------------------------
with open("wokwi_sim/serial_flameout.txt", "w") as f:
    for t in range(1, 251):
        if t < 90:
            temp = 35 + 45*(1-np.exp(-t/40))
            gas = 300
            phase = "Heating"
        elif t < 160:
            temp = 78 - 0.6*(t-90)
            gas = 300 + 2.2*(t-90)
            phase = "Flame Out"
        else:
            temp = 35 + 8*np.exp(-(t-160)/40)
            gas = 450 + 1.2*(t-160)
            phase = "Gas Leak"
        
        temp += np.random.randn()*1.2
        gas += np.random.randn()*6
        write_serial(f, t, temp, gas, True, "Flame-out", phase)

# --------------------------------------------------
# TIMEOUT (Person leaves kitchen)
# --------------------------------------------------
with open("wokwi_sim/serial_timeout.txt", "w") as f:
    for t in range(1, 301):
        temp = 45 + 25*(1-np.exp(-t/80))
        temp += np.random.randn()*1.5
        
        gas = 315 + 10*np.sin(t/40)
        gas += np.random.randn()*5
        
        if t < 70:
            presence = True
            phase = "Present"
        else:
            presence = False
            phase = "Unattended"
        
        write_serial(f, t, temp, gas, presence, "Timeout Risk", phase)

# --------------------------------------------------
# NORMAL COOKING (Typical cooking scenario)
# --------------------------------------------------
with open("wokwi_sim/serial_normal.txt", "w") as f:
    for t in range(1, 201):
        temp = 35 + 45*(1-np.exp(-t/60))
        temp += np.random.randn()*1.5
        
        gas = 300 + 15*np.sin(t/35)
        gas += np.random.randn()*4
        
        presence = np.random.rand() > 0.15
        phase = "Heating" if t < 60 else "Steady"
        
        write_serial(f, t, temp, gas, presence, "Normal Cooking", phase)

# --------------------------------------------------
# BOILOVER (Milk boiling over scenario)
# --------------------------------------------------
with open("wokwi_sim/serial_boilover.txt", "w") as f:
    for t in range(1, 201):
        # Rapid temperature rise to boiling
        temp = 30 + 70*(1 - np.exp(-t/25))
        temp += np.random.randn()*2.0
        
        # Gas fluctuates with boiling
        gas = 310 + 15*np.sin(t/20) + np.random.randn()*3
        
        # Presence is always true for boilover
        presence = True
        
        # Phase transitions
        if t < 40:
            phase = "Heating"
        elif t < 100:
            phase = "Approaching Boil"
        else:
            phase = "Boiling Over"
        
        write_serial(f, t, temp, gas, presence, "Milk Boilover", phase)

# --------------------------------------------------
# GAS LEAK (Slow gas leak with no flame)
# --------------------------------------------------
with open("wokwi_sim/serial_gasleak.txt", "w") as f:
    for t in range(1, 301):
        # Temperature stays near ambient
        temp = 28 + np.random.randn()*2 + 5*np.sin(t/150)
        
        # Gas slowly accumulates
        if t < 100:
            gas = 200 + 0.3*t + np.random.randn()*10
        elif t < 200:
            gas = 230 + 1.2*(t-100) + np.random.randn()*15
        else:
            gas = 350 + 2.5*(t-200) + np.random.randn()*20
        
        # Sometimes person is present, sometimes not
        presence = np.random.rand() > 0.4 if t < 150 else np.random.rand() > 0.7
        
        phase = "Gas Building" if t < 150 else "Dangerous Level"
        
        write_serial(f, t, temp, gas, presence, "Gas Leak", phase)

print("Created:")
print("  serial_falsealarm.txt")
print("  serial_sensornoise.txt")
print("  serial_returnuser.txt")
print("  serial_partialflameout.txt")
print("  serial_flameout.txt")
print("  serial_timeout.txt")
print("  serial_normal.txt")
print("  serial_boilover.txt")
print("  serial_gasleak.txt")
print("\n✅ All files use 7-field format: time|temp|gas|presence|scenario|phase|status")