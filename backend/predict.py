# predict.py
# ----------
# This file handles everything ML-related at serving time.
# "Serving time" means after the model is already trained —
# we are just using it to make predictions.
#
# One rule: NO database logic, NO API logic here.
# This file only knows about the ML model.
#
# Why separate this?
# If we ever swap Random Forest for a neural network,
# we only change THIS file. Nothing else needs to change.

import joblib
import pandas as pd

# ------------------------------------------------------------------
# STEP 1: Load the model once when this module is imported
# ------------------------------------------------------------------
# "When this module is imported" means: the moment another file
# writes "from predict import ..." — Python runs this file top to bottom.
# So the model loads exactly once, when the server starts.
# This is efficient — loading a model takes ~0.5 seconds.
# We never want to reload it on every request.

MODEL_PATH  = "model/bridgeclass_model.pkl"
FEATURES_PATH = "model/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print(f"✅ ML model loaded from {MODEL_PATH}")
print(f"✅ Features: {feature_names}")

# ------------------------------------------------------------------
# STEP 2: The prediction function
# ------------------------------------------------------------------
def predict_student_risk(
    attendance_rate: float,
    avg_grade: float,
    absences_last_month: int,
    gender: str,
    distance_km: float,
    num_siblings: int,
    fee_arrears: int
) -> dict:
    """
    Takes raw student data, returns a risk assessment.

    Args:
        All the student feature values as individual parameters.

    Returns:
        dict with keys:
            - risk_score (float 0.0-1.0)
            - risk_level (str: "low", "medium", "high")
            - top_risk_factors (list of human-readable strings)

    Why take individual parameters instead of a dict or object?
    Explicit parameters make the function self-documenting.
    You can see exactly what it needs without reading any other file.
    """

    # --- Encode gender ---
    # The model was trained with gender as 0 or 1, not "M" or "F"
    # We must apply the exact same encoding used in model.py
    gender_encoded = 0 if gender == "M" else 1

    # --- Build input DataFrame ---
    # The model expects a DataFrame with columns in a specific order.
    # feature_names is the list we saved in model.py — it defines that order.
    # If we pass columns in the wrong order, the model gives wrong predictions.
    input_data = pd.DataFrame([{
        "attendance_rate":      attendance_rate,
        "avg_grade":            avg_grade,
        "absences_last_month":  absences_last_month,
        "gender_encoded":       gender_encoded,
        "distance_km":          distance_km,
        "num_siblings":         num_siblings,
        "fee_arrears":          fee_arrears
    }])[feature_names]  # reorder columns to match training order exactly

    # --- Get risk probability ---
    # predict_proba returns a 2D array: [[prob_stayed, prob_dropout]]
    # [:, 1] takes the second column = probability of dropout
    # [0] takes the first (only) row
    risk_score = float(model.predict_proba(input_data)[0][1])

    # --- Classify into risk level ---
    risk_level = _classify_risk(risk_score)

    # --- Identify top risk factors ---
    # This explains the prediction in human language.
    # A teacher needs to know WHY a student is flagged, not just that they are.
    top_risk_factors = _get_risk_factors(
        attendance_rate, avg_grade,
        fee_arrears, distance_km,
        absences_last_month
    )

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "top_risk_factors": top_risk_factors
    }


# ------------------------------------------------------------------
# STEP 3: Helper functions (private)
# ------------------------------------------------------------------
def _classify_risk(score: float) -> str:
    """
    Converts probability score to a human-readable risk level.

    Thresholds:
    >= 0.6 → high   needs immediate teacher attention
    >= 0.3 → medium worth monitoring closely
    <  0.3 → low    currently on track
    """
    if score >= 0.6:
        return "high"
    elif score >= 0.3:
        return "medium"
    else:
        return "low"


def _get_risk_factors(
    attendance_rate: float,
    avg_grade: float,
    fee_arrears: int,
    distance_km: float,
    absences_last_month: int
) -> list[str]:
    """
    Builds a human-readable list of risk factors for a student.
    These explain the prediction to a teacher in plain language.

    We check each feature against a warning threshold.
    If a feature crosses its threshold, it becomes a risk factor.

    These thresholds are less strict than the dropout thresholds
    in seed.py — they are early warning signals, not confirmed risk.
    """
    factors = []

    if attendance_rate < 70:
        factors.append(
            f"Low attendance rate ({attendance_rate}%) — "
            f"threshold is 70%"
        )

    if avg_grade < 50:
        factors.append(
            f"Below average grade ({avg_grade}) — "
            f"threshold is 50"
        )

    if fee_arrears > 5000:
        factors.append(
            f"High fee arrears (KES {fee_arrears:,}) — "
            f"threshold is KES 5,000"
        )

    if distance_km > 15:
        factors.append(
            f"Long distance to school ({distance_km}km) — "
            f"threshold is 15km"
        )

    if absences_last_month > 8:
        factors.append(
            f"Many absences last month ({absences_last_month}) — "
            f"threshold is 8"
        )

    # If no single factor crossed a threshold but risk is still
    # elevated, the model detected a combined pattern
    if not factors:
        factors.append(
            "No single major factor — combined pattern detected by model"
        )

    return factors


def get_feature_names() -> list[str]:
    """
    Returns the feature names list.
    Used by database.py when seeding risk scores.
    """
    return feature_names


def get_model():
    """
    Returns the loaded model.
    Used by database.py when seeding risk scores.
    """
    return model