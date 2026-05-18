## Environment Setup
- Created conda environment: `bridgeclass` with Python 3.11
- Reason for 3.11: Full compatibility with scikit-learn, FastAPI, and pandas.
  Python 3.13 is too new for stable ML library support.
- Activated with: `conda activate bridgeclass`



## Week 1 — Day 1

### Step: Generated Synthetic Dataset (`seed.py`)

**What it does:**
Generates 1000 synthetic Kenyan student records using numpy and saves
them as a CSV file in `backend/data/students.csv`.

**Key decisions and why:**
- `np.random.seed(42)` — makes results reproducible. Same output every run.
- Used `np.random.normal()` for attendance and grades — bell curve matches
  real student performance distributions.
- Used `np.random.exponential()` for distance and fee arrears — right-skewed,
  meaning most values are small but some are very large. Matches real life.
- `.clip(0, 100)` — prevents impossible values like -5% attendance or 110 marks.

**Dropout label logic (grounded in Kenya education research):**
A student is marked dropped_out = 1 if ANY of these are true:
- attendance_rate < 55% (chronic absenteeism)
- avg_grade < 35 (academic failure threshold)
- fee_arrears > 12,000 KES (financial crisis)
- distance > 20km AND attendance < 70% (access barrier)

**Result:**
- 1000 records generated
- Dropout rate: 18.4% — matches Kenya's documented 18-22% range ✅
- Dataset saved to: `backend/data/students.csv`

**What I learned:**
- A DataFrame is a 2D table in pandas — like a spreadsheet in code
- Synthetic data must be calibrated against real-world statistics
- The dropout label (dropped_out) is called the TARGET VARIABLE —
  it is what the ML model will learn to predict



  ### EDA — Cell 5: Dropout Overview Charts

**Finding 1 — Class Imbalance:**
816 students stayed (81.6%) vs 184 dropped out (18.4%).
This 4:1 ratio confirms class imbalance exists.
Consequence: accuracy is NOT a valid metric for this model.
We will use F1-score and Recall instead.

**Finding 2 — Gender Gap:**
Female dropout rate: 19.3%
Male dropout rate:   17.5%
Girls drop out at a slightly higher rate — consistent with
Kenya education research (early marriage, domestic duties).
Gender will be included as a feature in the model.

**Plot saved:** `backend/plots/dropout_overview.png`


### EDA — Cell 6: Attendance & Grade Distributions

**Finding 3 — Attendance is the clearest separator:**
Students below 55% attendance are almost exclusively in the
dropout group. The dashed threshold line shows a clean split.
This will likely be the model's most important feature.

**Finding 4 — Grade overlap is real:**
Below 35 average grade = strong dropout signal.
Between 35-60 = mixed zone where other features (fees, distance)
must help the model decide. This is why ML is necessary —
simple threshold rules miss the nuanced cases in the overlap.

**Key insight for interviews:**
"The overlap between stayed and dropped-out students in the
35-60 grade range shows that no single feature is sufficient.
The model learns combinations of features — low attendance
AND moderate grades AND high fee arrears together signal
risk better than any one feature alone."

**Plot saved:** `backend/plots/attendance_grade_distribution.png`

### EDA — Cell 7: Correlation Matrix

**Correlations with dropout_out (key findings):**
- attendance_rate: -0.38 → strongest negative predictor
- avg_grade:       -0.35 → second strongest negative predictor  
- fee_arrears:     +0.18 → positive predictor (more debt = more risk)
- distance_km:     +0.04 → near zero LINEAR correlation

**Critical insight — why correlation alone is not enough:**
distance_km shows near-zero correlation with dropout individually.
But in seed.py, the dropout condition was:
distance > 20km AND attendance < 70% (combined condition).
Linear correlation cannot detect feature interactions.
Random Forest can — it evaluates combinations at every tree split.
This justifies using ML over simple statistical thresholds.

**What I would say in an interview:**
"The correlation matrix showed attendance and grades as the
strongest individual predictors. However, distance_km showed
near-zero correlation despite being part of the dropout logic.
This is because its effect is conditional — it only matters
when combined with low attendance. Random Forest captures
these interaction effects, which is one reason I chose it
over logistic regression."

**Plot saved:** `backend/plots/correlation_matrix.png`

### EDA — Cell 8: Box Plots (Fee Arrears & Distance)

**Finding 5 — Fee arrears median is higher for dropouts:**
The dropout group's fee arrears box sits higher overall,
with several outliers above 12,000 KES — these are students
whose families are in genuine financial crisis.

**Finding 6 — Distance boxes are nearly identical:**
Stayed vs Dropped Out show almost the same distance
distribution. Confirms that distance has no standalone
predictive power — its effect is conditional on attendance.
Visual proof of why the correlation was only 0.04.

**=== Full EDA Summary ===**
Top predictors of dropout (from correlation analysis):
1. attendance_rate (-0.38) — strongest signal
2. avg_grade (-0.35) — second strongest
3. fee_arrears (+0.18) — financial pressure
4. distance_km (+0.04) — conditional, not standalone

Class imbalance confirmed: 81.6% stayed vs 18.4% dropped out.
Must use F1-score and Recall as evaluation metrics, not accuracy.

All 4 plots saved to backend/plots/


## Week 1 — Day 2

### Step: Trained Random Forest Model (`model.py`)

**Results:**
- Accuracy:  100.0%
- Precision: 100.0%
- Recall:    100.0%
- F1-Score:  100.0%
- ROC-AUC:   1.000
- Cross-val Mean F1: 0.983 (+/- 0.016)

**Why 100% is a red flag, not a celebration:**
Our synthetic data uses hard threshold rules to generate dropout
labels (e.g. attendance < 55% = dropout). The Random Forest
learned these exact thresholds perfectly — essentially reverse-
engineering the rules we wrote in seed.py.

Real-world data has noise — a student at 54% attendance who
stayed due to family support, or a 57% attender who dropped out
for an unrelated reason. Our synthetic data has no such noise,
making 100% achievable but not realistic.

The cross-validation scores are more honest:
[1.0, 1.0, 0.972, 0.959, 0.986] — two folds dropped to ~96-97%,
showing the model is strong but not magically perfect.

**What I would say in an interview:**
"The model achieved 100% on the test set, which I recognized
as a symptom of the synthetic data's clean threshold logic
rather than a genuinely perfect model. Cross-validation showed
a more realistic mean F1 of 0.983 with variance across folds.
With real school data containing natural noise, I would expect
F1 in the 0.75-0.85 range, which is still strong for this task."

**Feature Importance (most important finding):**
1. attendance_rate    40.6% — strongest driver
2. avg_grade          38.3% — second strongest
3. fee_arrears        11.6% — financial pressure
4. distance_km         4.7% — conditional effect
5. absences_last_month 2.7%
6. num_siblings        1.6%
7. gender_encoded      0.5%

Attendance + grades together drive 78.9% of all model decisions.
This aligns with published Kenya education research — strong
external validity for a synthetic dataset.

**Files saved:**
- model/bridgeclass_model.pkl  ← trained model
- model/feature_names.pkl      ← feature order for API
- plots/confusion_matrix.png
- plots/roc_curve.png
- plots/feature_importance.png


