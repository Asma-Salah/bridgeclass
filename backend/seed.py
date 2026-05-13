# seed.py
# -------
# This script generates a synthetic dataset of 1000 Kenyan students.
# "Synthetic" means fake but realistic — we design the numbers to reflect
# real-world patterns documented in Kenya education research.
# This removes any dependency on real school data for our prototype.

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------
# STEP 1: Set a random seed
# ------------------------------------------------------------------
# np.random.seed() makes our random numbers reproducible.
# Every time this script runs, it produces the EXACT same dataset.
# Without this, the numbers would be different every run — 
# making it impossible to compare results consistently.
np.random.seed(42)

# ------------------------------------------------------------------
# STEP 2: Define how many students to generate
# ------------------------------------------------------------------
N = 1000  # 1000 student records

# ------------------------------------------------------------------
# STEP 3: Generate each feature column
# ------------------------------------------------------------------

# --- Attendance Rate (0 to 100%) ---
# np.random.normal(mean, std_deviation, count)
# Most students attend ~75% of the time, with variation of ±15%
# .clip(0, 100) ensures no value goes below 0 or above 100
attendance_rate = np.random.normal(75, 15, N).clip(0, 100)

# --- Average Grade (0 to 100) ---
# Most students score around 58%, std of 18
avg_grade = np.random.normal(58, 18, N).clip(0, 100)

# --- Absences Last Month (whole numbers) ---
# randint gives whole numbers between 0 and 14
absences_last_month = np.random.randint(0, 15, N)

# --- Gender ---
# np.random.choice picks randomly from a list
# In Kenya, schools are roughly 50/50 gender split
gender = np.random.choice(["M", "F"], N)

# --- Distance to School (km) ---
# np.random.exponential gives a right-skewed distribution —
# most students live close, but some live very far
# This matches real rural Kenya geography
distance_km = np.random.exponential(scale=5, size=N).clip(0, 40)

# --- Number of Siblings ---
# Kenya has a higher average household size than Western countries
# Most students have 2-4 siblings
num_siblings = np.random.randint(0, 8, N)

# --- Fee Arrears (unpaid school fees in KES) ---
# Many families owe some fees — exponential distribution again
# because most owe little but some owe a lot
fee_arrears = np.random.exponential(scale=3000, size=N).clip(0, 20000)

# --- Generate realistic Kenyan student names ---
first_names = [
    "Amina", "Brian", "Cynthia", "David", "Esther", "Francis",
    "Grace", "Hassan", "Irene", "James", "Kezia", "Linet",
    "Moses", "Naomi", "Oliver", "Purity", "Quinter", "Robert",
    "Sharon", "Thomas", "Usha", "Vincent", "Wanjiku", "Xavier",
    "Yvonne", "Zainab", "Aisha", "Boniface", "Caroline", "Dennis"
]
last_names = [
    "Osei", "Kamau", "Otieno", "Waweru", "Muthoni", "Kariuki",
    "Odhiambo", "Njoroge", "Achieng", "Mwangi", "Koech", "Rotich",
    "Simiyu", "Chebet", "Mutua", "Njenga", "Owino", "Auma",
    "Wekesa", "Nyambura", "Juma", "Moraa", "Kibet", "Adhiambo"
]

# Generate N random full names by combining random first + last names
names = [
    f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
    for _ in range(N)
]

# ------------------------------------------------------------------
# STEP 4: Assemble into a DataFrame
# ------------------------------------------------------------------
# A DataFrame is like a spreadsheet in Python — rows and columns
df = pd.DataFrame({
    "name": names,
    "gender": gender,
    "attendance_rate": attendance_rate.round(1),
    "avg_grade": avg_grade.round(1),
    "absences_last_month": absences_last_month,
    "distance_km": distance_km.round(1),
    "num_siblings": num_siblings,
    "fee_arrears": fee_arrears.round(0).astype(int),
})

# ------------------------------------------------------------------
# STEP 5: Generate the dropout label
# ------------------------------------------------------------------
# This is the TARGET variable — what our model will learn to predict.
# We define dropout based on realistic risk thresholds:
# A student is marked as dropped out (1) if ANY of these are true:
#   - attendance below 55% (chronic absenteeism)
#   - average grade below 35 (academic failure)
#   - fee arrears above 12,000 KES (financial crisis)
#   - distance over 20km AND attendance below 70% (access barrier)
#
# Otherwise they stayed (0).
# This logic is grounded in Kenya education research.

dropout_condition = (
    (df["attendance_rate"] < 55) |
    (df["avg_grade"] < 35) |
    (df["fee_arrears"] > 12000) |
    ((df["distance_km"] > 20) & (df["attendance_rate"] < 70))
)

df["dropped_out"] = dropout_condition.astype(int)

# ------------------------------------------------------------------
# STEP 6: Check the dropout rate
# ------------------------------------------------------------------
# Kenya's actual secondary dropout rate is ~18-22%
# Let's verify our synthetic data is in that range
dropout_rate = df["dropped_out"].mean() * 100
print(f"Generated {N} student records.")
print(f"Dropout rate: {dropout_rate:.1f}%  (target: 18-25%)")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn summary:")
print(df.describe())

# ------------------------------------------------------------------
# STEP 7: Save to CSV
# ------------------------------------------------------------------
# os.path.join builds a file path that works on any OS
output_path = os.path.join("data", "students.csv")
df.to_csv(output_path, index=False)
# index=False means don't save the row numbers as a column

print(f"\nDataset saved to {output_path}")