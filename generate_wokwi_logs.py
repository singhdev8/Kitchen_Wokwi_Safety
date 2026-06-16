import numpy as np

# --------------------------------------------------
# FALSE ALARM
# --------------------------------------------------

with open("wokwi_sim/serial_falsealarm.txt", "w") as f:
    for t in range(1, 201):

        temp = 70 + 15*np.sin(t/30) + np.random.randn()*1.5
        gas  = 320 + 20*np.sin(t/25) + np.random.randn()*5

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | YES | SAFE\n"
        )

# --------------------------------------------------
# SENSOR NOISE
# --------------------------------------------------

with open("wokwi_sim/serial_sensornoise.txt", "w") as f:

    for t in range(1, 201):

        temp = 65 + np.random.randn()*8
        gas  = 300 + np.random.randn()*30

        if np.random.rand() < 0.05:
            temp += np.random.randint(15, 35)

        if np.random.rand() < 0.05:
            gas += np.random.randint(80, 150)

        f.write(
            f"{t}s | {temp:.1f}C | {max(gas,0):.0f}ppm | YES | SAFE\n"
        )

# --------------------------------------------------
# RETURN USER
# --------------------------------------------------

with open("wokwi_sim/serial_returnuser.txt", "w") as f:

    for t in range(1, 201):

        temp = 55 + 10*np.sin(t/40) + np.random.randn()*2
        gas  = 310 + 10*np.sin(t/30)

        if t < 60:
            pir = "YES"
        elif t < 140:
            pir = "NO"
        else:
            pir = "YES"

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | {pir} | SAFE\n"
        )

# --------------------------------------------------
# PARTIAL FLAMEOUT
# --------------------------------------------------

with open("wokwi_sim/serial_partialflameout.txt", "w") as f:

    for t in range(1, 201):

        if t < 70:

            temp = 80 - 0.05*t

            gas = 300

            status = "SAFE"

        elif t < 120:

            temp = 76 - 0.55*(t-70)

            gas = 300 + 2*(t-70)

            status = "SAFE"

        elif t < 150:

            temp = 48 + 0.4*(t-120)

            gas = 390

            status = "SAFE"

        else:

            temp = 60 - 0.45*(t-150)

            gas = 390 + 1.5*(t-150)

            status = "SAFE"

        temp += np.random.randn()*1.0
        gas  += np.random.randn()*5

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | YES | {status}\n"
        )
# --------------------------------------------------
# FLAMEOUT (REPLACEMENT)
# --------------------------------------------------

with open("wokwi_sim/serial_flameout.txt", "w") as f:

    for t in range(1, 251):

        if t < 90:

            temp = 35 + 45*(1-np.exp(-t/40))
            gas = 300

        elif t < 160:

            temp = 78 - 0.6*(t-90)
            gas = 300 + 2.2*(t-90)

        else:

            temp = 35 + 8*np.exp(-(t-160)/40)
            gas = 450 + 1.2*(t-160)

        temp += np.random.randn()*1.2
        gas += np.random.randn()*6

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | YES | SAFE\n"
        )

# --------------------------------------------------
# TIMEOUT (REPLACEMENT)
# --------------------------------------------------

with open("wokwi_sim/serial_timeout.txt", "w") as f:

    for t in range(1, 301):

        temp = 45 + 25*(1-np.exp(-t/80))
        temp += np.random.randn()*1.5

        gas = 315 + 10*np.sin(t/40)
        gas += np.random.randn()*5

        if t < 70:
            pir = "YES"
        else:
            pir = "NO"

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | {pir} | SAFE\n"
        )

# --------------------------------------------------
# NORMAL COOKING (REPLACEMENT)
# --------------------------------------------------

with open("wokwi_sim/serial_normal.txt", "w") as f:

    for t in range(1, 201):

        temp = 35 + 45*(1-np.exp(-t/60))
        temp += np.random.randn()*1.5

        gas = 300 + 15*np.sin(t/35)
        gas += np.random.randn()*4

        pir = "YES" if np.random.rand() > 0.15 else "NO"

        f.write(
            f"{t}s | {temp:.1f}C | {gas:.0f}ppm | {pir} | SAFE\n"
        )                


print("Created:")
print("  serial_falsealarm.txt")
print("  serial_sensornoise.txt")
print("  serial_returnuser.txt")
print("  serial_partialflameout.txt")