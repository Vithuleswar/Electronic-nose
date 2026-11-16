# Low-cost electronic nose: ML smell classification with Arduino Uno + MQ-135 / MQ-3 / MQ-6

Three cross-sensitive metal-oxide gas sensors act as a tiny "nose". Each odour produces a different
pattern across the sensors. The Arduino streams the raw readings, Python turns them into
baseline-normalised features, and a classifier recognises which odour is present.

```
MQ-135, MQ-3, MQ-6 --> Arduino Uno (16x oversampled ADC, 5 Hz) --USB--> Python
   Rs/R0 normalisation -> 20 s sliding windows -> 22 features -> classifier -> live prediction
```

## 1. Hardware

| Part | Qty |
|---|---|
| Arduino Uno + USB-B cable | 1 |
| Breadboard + jumper wires | 1 |
| MQ-135, MQ-3, MQ-6 modules (4 pins: VCC, GND, DO, AO) | 1 each |
| Laptop | 1 |
| Clear plastic cup / cut bottle (the "hood"), cotton balls, bottle caps | a few |

### Wiring

| Module pin | Goes to |
|---|---|
| MQ-135 VCC / GND / **AO** | breadboard + rail (5V) / - rail (GND) / **A0** |
| MQ-3 VCC / GND / **AO** | breadboard + rail (5V) / - rail (GND) / **A1** |
| MQ-6 VCC / GND / **AO** | breadboard + rail (5V) / - rail (GND) / **A2** |
| DO (all three) | not connected |
| Arduino 5V -> + rail, Arduino GND -> - rail | one jumper each |

* Pin order differs between module brands. **Read the silkscreen labels** (VCC, GND, DO, AO) and check polarity before plugging in USB.
* The three heaters draw roughly 150 mA each (about 0.45 A) plus the Arduino itself: right at the USB 2.0 limit.
  Plug directly into the laptop (a USB 3.0 port is best), not into an unpowered hub. If the board resets or readings sag, power the
  sensor rail from an external 5 V supply of at least 1 A and join its GND to the Arduino GND.
* The potentiometer on each module only sets the DO threshold. We use the analog output, so ignore it.

## 2. Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt   # note: the package is "pyserial", NOT "serial"
```

1. Arduino IDE: open `firmware/enose_firmware/enose_firmware.ino` -> Tools -> Board: *Arduino Uno* -> choose the Port -> Upload.
2. Tools -> Serial Monitor at **115200 baud**: you should see `ms,mq135,mq3,mq6` and four columns of numbers. **Close the Serial Monitor afterwards.**
3. **Burn-in:** leave the sensors powered for 48 h the first time (the Hanwei datasheets ask for more than 24 h of preheat, and one MQ-3 vendor recommends more than 48 h). Before every later session allow at least 20 minutes of warm-up.
4. Test the software today without hardware (SYNTHETIC data, pipeline test only):
   ```bash
   python simulate_data.py
   python train_model.py --data data/synthetic --out outputs_synthetic
   python live_predict.py --simulate --model outputs_synthetic/enose_model.joblib
   ```
5. **Screen your three items first** (about 30 minutes, one session). It shows whether the items are strong and different from each other *before* you spend days recording:
   ```bash
   python collect_data.py --session SCREEN --labels clean_air incense_stick camphor perfume --repeats 2 --baseline 45 --expose 45 --purge 90 --out data/screen
   python screen_summary.py --data data/screen
   ```
   Read the table and `outputs_screen/fingerprints.png`. Drop any class marked WEAK. For each CLOSE pair keep only the stronger, more repeatable class, or keep both and report the confusion.
6. Record real data, at least **2 sessions on different days** (3 is better; about 12 runs, 1 h per session with 4 classes):
   ```bash
   python collect_data.py --session S1 --labels clean_air incense_stick camphor perfume --repeats 3
   python collect_data.py --session S2 --labels clean_air incense_stick camphor perfume --repeats 3
   python collect_data.py --session S3 --labels clean_air incense_stick camphor perfume --repeats 3
   ```
   Interrupted? Resume with `--start-at 9` (same session name).
7. Train and evaluate: `python train_model.py`  (results in `outputs/`)
8. Live demo: `python live_predict.py`

## 3. Measurement protocol (keep it identical every run)

Setup: breadboard on a flat surface, sensors facing up, the "hood" (inverted cup) over them with two coins under the rim so air can leak in slowly.
Mark the spot where the sample cap goes (about 3 cm from the sensors). Keep sample bottles and spice jars **closed and in another room**: open containers contaminate the baseline.
**Record in a quiet room away from the stove, never while anyone is cooking:** cooking fumes and gas register on all three sensors.
**The MQ-3 is an alcohol specialist and reacts strongly to ethanol vapour.** No hand sanitiser, perfume, deodorant or aftershave on you or your clothes during a session, and keep the perfume sample sealed away between runs. Seal the camphor and incense between runs too: open samples contaminate the baseline.

| Phase | What you do | Default length |
|---|---|---|
| BASELINE | hood on, clean air, do not breathe on the sensors | 60 s or more |
| EXPOSURE | lift hood, place the sample cap, lower hood, press ENTER | 75 s |
| PURGE | lift hood, take the sample away, fan the air | 150 s |

`collect_data.py` shuffles the run order, beeps at each phase change, warns if the baseline is still drifting, and tells you the peak response of each run.
Use `clean_air` for **blank runs**: do every handling step with a cap holding a cotton ball wetted with 1 teaspoon of plain water.
Some samples are damp (perfume) and MQ sensors react to humidity, so the blank must be humidity-matched. Blanks teach the model what "just handling" looks like.

Sample amounts (the same every time). Pick one chemical family per class so each leans on a different sensor:

| Label | Kitchen item and amount | Family | Expected main sensor |
|---|---|---|---|
| `clean_air` | cap with a cotton ball and 1 tsp plain water (blank) | control | none |
| `incense_stick` | light one stick, blow out the flame, let it smoulder 10 s, hold the smoking tip ~10 cm from the sensors for 5 s, dip it in water | combustion smoke + fragrance | MQ-135 (+ some MQ-3) |
| `camphor` | 1 pea-sized piece (pooja tablet) in the cap | terpenoid vapour | MQ-135 |
| `perfume` | 2 sprays on a cotton ball in the cap (never spray the sensors) | ethanol + fragrance | MQ-3 and MQ-135 |

The "expected sensor" column is a prediction from the datasheets. The screening step replaces the prediction with your own measurements.

## 4. How the ML works (and why it is built this way)

* **Normalisation.** For a module with load resistor RL, `Rs/RL = (1023 - adc)/adc`. RL cancels in `Rs/R0`, so we never need it. The response is
  `y = -ln(Rs/R0)`, with R0 measured in clean air at the start of every run and live session. This removes most sensor-to-sensor and day-to-day offset.
* **Windows and features.** 20 s windows every 2 s; per sensor: mean, 95th percentile, std, slope, start-to-end change, last value; plus total response and
  each sensor's share of it (a concentration-independent fingerprint). 22 features in all. The slope features capture the *kinetics*, which separate odours that look alike at steady state.
* **Labels.** Baseline windows -> `clean_air`; windows ending 5 s or more into the exposure -> the sample; purge windows once the response has decayed -> `clean_air`; windows during recovery are dropped.
* **Models compared:** logistic regression, kNN, SVM (RBF), random forest, histogram gradient boosting, small MLP. The best by macro-F1 is saved.
* **Honest validation.** Windows from one run overlap in time and share the baseline, so a random split leaks. Sensor drift can also correlate with the order in which samples were recorded, which is why
  the order is shuffled. Training therefore holds out whole **sessions** (leave-one-session-out; with 2 sessions that means train on one and test on the other, then swap; whole runs only if you have a single session). The script prints the naive random-split score
  next to the honest one so the gap is visible. **Headline metrics are macro-F1 and balanced accuracy**, because clean-air windows dominate plain accuracy.
* **Class selection by screening.** `screen_summary.py` measures each candidate's peak response, fingerprint (which sensor carries it) and rise time, compares them with your blank-run noise floor, and lists the closest pairs. Classes are chosen from data, not guessed.
* **Ablation.** Static features only vs. static + kinetic features vs. without the fingerprint shares.
* **Uncertainty.** `live_predict.py` shows "uncertain" when the top probability is below 0.60 instead of forcing a guess.

## 5. Files

| File | Purpose |
|---|---|
| `firmware/enose_firmware/enose_firmware.ino` | Arduino: reads A0-A2, averages 16 readings, prints CSV at 5 Hz |
| `enose_features.py` | normalisation, windowing, features (shared by training and live use) |
| `enose_io.py` | serial reader thread, simulator reader, CSV writer |
| `collect_data.py` | guided, randomised data collection |
| `train_model.py` | model comparison, grouped CV, ablation, plots, saves the model |
| `live_predict.py` | real-time classification with a live plot |
| `screen_summary.py` | screening report: fingerprints, strength vs. blank noise, closest (most confusable) pairs |
| `simulate_data.py` | SYNTHETIC data for pipeline testing only |

`screen_summary.py` writes `fingerprints.png` and `fingerprints.csv` to `outputs_screen/`. `train_model.py` writes to `outputs/`: `enose_model.joblib`, `metrics.json`, `model_comparison.csv`, `confusion_matrix.png`, `pca_windows.png`, `feature_importance.png`.

## 6. Safety

* The sensor heaters get hot. Do not touch the metal mesh while powered and keep flammables away from it.
* **Incense:** keep the smoking tip out from under the plastic hood and away from paper and cloth, and put it out in water after each run. Ventilate the room afterwards.
* **Camphor** is flammable and strongly scented: one small piece, no flames or sparks, ventilated room, wash your hands afterwards. Never eat pooja camphor and keep it away from children.
* **Perfume** is alcohol-based and flammable: 2 sprays, no flames or sparks.
* Do not breathe concentrated vapours or smoke. Do not mix cleaning products (for example ammonia cleaners with bleach). Skip any class you are not comfortable handling.
* This is a teaching prototype, **not** a certified gas or smoke detector. Never rely on it for safety.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| No serial port appears | Use a data-capable USB cable. Clone boards (CH340 chip) need the CH340 driver. |
| "Could not open port" | Close the Arduino Serial Monitor / Plotter. Only one program may use the port. |
| Arduino resets or readings sag when sensors are connected | Power budget, see section 1. Try another USB port or an external 5 V supply. |
| A channel reads ~0 or ~1023, or is perfectly flat | Check VCC / GND / AO wiring and polarity. |
| `ModuleNotFoundError: serial` | `pip uninstall serial` then `pip install pyserial`. |
| Baseline keeps drifting | Warm up longer, keep drafts, fans, AC and your breath away, record in the same room each time. |
| Almost no response | Sample too small or too far, sensors still cold, or a blank run. |
| MQ-3 baseline jumps for no reason | Ethanol on your hands or clothes, an open perfume bottle in the room, or a draft. Wash hands, seal samples, wait. |
| A channel sits very low (< 20) or very high (> 1000) in clean air | The module's load resistor is far from the sensor's resistance, so resolution is coarse (16x averaging helps). Usable; note it in the report. |
| Signal goes DOWN when a sample is added | Unusual module type; this code assumes AO rises with gas. |
| The Arduino restarts every time a script starts | Normal: opening the port resets the Uno. The sensors stay hot. |

## 8. Results (fill in from `outputs/metrics.json`; use your REAL data only)

| Model | macro-F1 (leave-one-session-out) | balanced accuracy |
|---|---|---|
| | | |

Report: the confusion matrix, per-session scores, the ablation table, the naive-vs-honest gap, and which pairs are confused and why (that is a finding, not a failure).

## 9. Limitations and extensions

* Three sensors identify an odour *fingerprint*, not a molecule. The array is one broad sensor (MQ-135: VOCs, ammonia, smoke) plus two specialists (MQ-3: alcohol; MQ-6: LPG, butane, propane). With these items only perfume (ethanol) should stand out on the MQ-3, and the MQ-6 will mostly stay quiet because none of the items is LPG or butane. Incense smoke and camphor land mainly on the MQ-135, so they are told apart by strength, by the secondary sensors and by kinetics. The screening step exists to find any overlap between them.
* No temperature or humidity sensing; both change MQ readings. Extension: add a DHT22 and use T/RH as extra features.
* Manual sample delivery limits repeatability. Extension: a small fan or pump and a fixed chamber.
* Extensions: probability calibration, novelty detection for unseen odours, drift compensation across sessions, on-device inference with a small decision tree, a food-freshness case study.

## 10. Background reading (verify and format in IEEE style for the report)

* N. Dennler et al., "Drift in a popular metal oxide sensor dataset reveals limitations for gas classification benchmarks," *Sensors and Actuators B: Chemical*, vol. 361, 131668, 2022.
* Jaya Z., "Perfume classification using a low-cost electronic nose with Isolation Forest and Linear Discriminant Analysis," *IEEE AIMS*, 2025.
* "Using a low-cost components e-nose for basic detection of different foodstuffs," IEEE Xplore, document 9799799.
* H. K. Salih and A. Al-Ani, "Array of Gas Sensors Data" (odours: air, onion, garlic, lemon, tomato, petrol, gasoline, coffee, orange, perfume), IEEE DataPort.
* "Classification of Flavored Filipino Vinegars Using Electronic Nose" (MQ sensors), MDPI Engineering Proceedings, vol. 92, no. 1, art. 16.
* Hanwei MQ-3, MQ-6 and MQ-135 datasheets.
#   E l e c t r o n i c - n o s e  
 #   E l e c t r o n i c - n o s e  
 