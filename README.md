# Smart Kitchen Safety System

### Hybrid Rule-Based and Machine Learning Approach with Predictive Hazard Detection

---

## Project Overview

This repository contains supplementary information for the research paper:

**"Smart Kitchen Safety System: Hybrid Rule-Based and Machine Learning Approach with Predictive Hazard Detection."**

The project presents a **three-layer intelligent kitchen safety framework** that combines deterministic safety rules, machine learning, and predictive analytics to detect hazardous situations before they become critical.

Unlike conventional kitchen safety systems that react only after a dangerous event has occurred, the proposed system provides **early warnings** and **automatic protective actions** using environmental sensor data.

---

## Research Objectives

The proposed framework aims to:

* Detect LPG gas leakage.
* Predict milk/food boil-over before overflow occurs.
* Identify flame-out conditions.
* Detect prolonged unattended cooking.
* Reduce household kitchen accidents through intelligent decision making.
* Provide early warning instead of only reactive protection.

---

## System Architecture

The system consists of three cooperative decision layers.

### Layer 1 — Rule-Based Safety Engine

A deterministic safety engine continuously monitors sensor values and immediately performs protective actions whenever predefined safety conditions are satisfied.

Detected hazards include:

* Boil-over
* Gas leakage
* Flame-out
* Timeout due to prolonged absence

This layer is responsible for guaranteed safety shutdown.

---

### Layer 2 — Machine Learning Classifier

A Random Forest model analyzes multiple sensor features simultaneously to recognize cooking conditions.

The classifier categorizes the environment into:

* Normal Cooking
* Boil-over Risk
* Gas Leak
* Timeout Risk
* Flame-out

Instead of relying on fixed thresholds, this layer learns complex relationships among sensor readings to recognize hazardous situations earlier.

---

### Layer 3 — Predictive Forecasting

The forecasting layer estimates future system behaviour using recent sensor trends.

It provides:

* Early boil-over prediction
* Burn-risk estimation
* Forgetfulness probability estimation
* Future temperature forecasting

This enables proactive alerts before dangerous conditions actually occur.

---

## Hardware Platform

The embedded implementation was validated using an ESP32-based simulation.

The virtual prototype includes:

* ESP32
* Temperature sensor
* LPG gas sensor
* PIR motion sensor
* Relay module
* Buzzer
* 16×2 LCD display
* User control buttons

Simulation and firmware validation were performed using the **Wokwi** platform.

---

## Key Contributions

* Hybrid Rule-Based + Machine Learning architecture
* Three-layer intelligent safety framework
* Predictive hazard detection
* Early warning mechanism
* Automatic relay shutdown
* Embedded ESP32 implementation
* Real-time simulation validation
* Lightweight architecture suitable for IoT deployment

---

## Experimental Highlights

The proposed framework achieved:

* Approximately **97% classification accuracy**
* Gas leak detected **25 seconds earlier** than conventional threshold detection
* Boil-over predicted approximately **107 seconds in advance**
* Five kitchen scenarios successfully validated

These results demonstrate the effectiveness of combining rule-based reasoning with machine learning and predictive forecasting.

---

## Repository Contents

This supplementary repository intentionally **does not include the implementation source code**.

It provides documentation describing the proposed architecture, methodology, and research contribution while protecting the underlying implementation.

---

## Citation

If you use or reference this work, please cite:

**Devkaran Singh Sarkaria**

*Smart Kitchen Safety System: Hybrid Rule-Based and Machine Learning Approach with Predictive Hazard Detection.*

Punjab Engineering College, Chandigarh, India.

---

## Contact

**Devkaran Singh Sarkaria**

Punjab Engineering College (PEC)
Chandigarh – 160012, India

📧 Email: [d522kss@gmail.com](mailto:d522kss@gmail.com)

📞 Phone: +91 7973820938

---

  ## NOTE

The implementation source code is not included in this supplementary material.

This repository is intended solely to provide supporting documentation for the associated research paper. The software implementation remains the intellectual property of the author and may be shared separately for academic collaboration or research purposes upon request.
