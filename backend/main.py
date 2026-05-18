# main.py
# -------
# This is the FastAPI backend — the bridge between the
# React frontend and the ML model.
#
# It exposes HTTP endpoints that the frontend calls to:
# 1. Get all students with their risk scores
# 2. Get one student's full profile
# 3. Predict risk for a new student

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import sqlite3
import os

# ------------------------------------------------------------------
# STEP 1: Initialize the FastAPI app
# ------------------------------------------------------------------
app = FastAPI(
    title="BridgeClass API",
    description="AI-powered student dropout early-warning system",
    version="1.0.0"
)

# ------------------------------------------------------------------
# STEP 2: CORS Middleware
# ------------------------------------------------------------------
# CORS = Cross Origin Resource Sharing
# When React (port 3000) talks to FastAPI (port 8000),
# the browser blocks it by default — different ports = different origins.
# This middleware tells FastAPI: "allow requests from React dev server."
# Without this, every API call from the frontend would fail silently.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# STEP 3: Load the trained ML model on startup
# ------------------------------------------------------------------
# We load the model ONCE when the server starts — not on every request.
# Loading a model takes time. Loading it once and reusing it is efficient.

MODEL_PATH = "model/bridgeclass_model.pkl"
FEATURES_PATH = "model/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"✅ Features: {feature_names}")

# ------------------------------------------------------------------
# STEP 4: Database setup
# ------------------------------------------------------------------
# SQLite is a file-based database — no server needed.
# The database file is created automatically if it doesn't exist.

DB_PATH = "bridgeclass.db"

def get_db():
    """Create a database connection. Used by every endpoint that needs data."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # returns rows as dictionaries, not tuples
    return conn

def init_db():
    """
    Create the students table and populate it from our CSV.
    This runs once when the server starts.
    If the table already exists, it does nothing.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Create table if it doesn't exist yet
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            attendance_rate REAL,
            avg_grade REAL,
            absences_last_month INTEGER,
            distance_km REAL,
            num_siblings INTEGER,
            fee_arrears INTEGER,
            dropped_out INTEGER,
            risk_score REAL,
            risk_level TEXT
        )
    """)

    # Only seed data if table is empty
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]

    if count == 0:
        print("⏳ Seeding database from CSV...")
        df = pd.read_csv("data/students.csv")

        # Compute risk scores for every student using our trained model
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["gender_encoded"] = le.fit_transform(df["gender"])

        X = df[feature_names]
        risk_scores = model.predict_proba(X)[:, 1]  # probability of dropout
        df["risk_score"] = risk_scores.round(4)

        # Classify into risk levels
        def classify_risk(score):
            if score >= 0.6:
                return "high"
            elif score >= 0.3:
                return "medium"
            else:
                return "low"

        df["risk_level"] = df["risk_score"].apply(classify_risk)

        # Insert all students into the database
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO students
                (name, gender, attendance_rate, avg_grade,
                 absences_last_month, distance_km, num_siblings,
                 fee_arrears, dropped_out, risk_score, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["name"], row["gender"],
                row["attendance_rate"], row["avg_grade"],
                row["absences_last_month"], row["distance_km"],
                row["num_siblings"], row["fee_arrears"],
                row["dropped_out"], row["risk_score"], row["risk_level"]
            ))

        conn.commit()
        print(f"✅ Database seeded with {len(df)} students")
    else:
        print(f"✅ Database already has {count} students")

    conn.close()

# Run database setup immediately when server starts
init_db()

# ------------------------------------------------------------------
# STEP 5: Pydantic Models (Data Schemas)
# ------------------------------------------------------------------
# Pydantic models define the SHAPE of data coming in and going out.
# Think of them as TypeScript interfaces but in Python.
# FastAPI uses them to automatically validate all requests.

class StudentInput(BaseModel):
    """Data required to predict risk for a NEW student."""
    name: str
    gender: str = Field(..., pattern="^[MF]$")
    attendance_rate: float = Field(..., ge=0, le=100)
    avg_grade: float = Field(..., ge=0, le=100)
    absences_last_month: int = Field(..., ge=0)
    distance_km: float = Field(..., ge=0)
    num_siblings: int = Field(..., ge=0)
    fee_arrears: int = Field(..., ge=0)

class PredictionResponse(BaseModel):
    """Data returned after a risk prediction."""
    name: str
    risk_score: float
    risk_level: str
    top_risk_factors: list[str]

# ------------------------------------------------------------------
# STEP 6: API Endpoints
# ------------------------------------------------------------------

@app.get("/")
def root():
    """Health check — confirms the API is running."""
    return {
        "message": "BridgeClass API is running",
        "version": "1.0.0",
        "endpoints": ["/students", "/students/{id}", "/predict", "/stats"]
    }


@app.get("/students")
def get_all_students(risk_level: str = None, limit: int = 100):
    """
    Get all students with their risk scores.
    Optional filter: ?risk_level=high OR ?risk_level=medium OR ?risk_level=low
    Optional limit: ?limit=50
    """
    conn = get_db()
    cursor = conn.cursor()

    if risk_level:
        # Filter by risk level if provided
        cursor.execute("""
            SELECT * FROM students
            WHERE risk_level = ?
            ORDER BY risk_score DESC
            LIMIT ?
        """, (risk_level, limit))
    else:
        # Return all students, highest risk first
        cursor.execute("""
            SELECT * FROM students
            ORDER BY risk_score DESC
            LIMIT ?
        """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    # Convert SQLite rows to regular Python dictionaries
    return [dict(row) for row in rows]


@app.get("/students/{student_id}")
def get_student(student_id: int):
    """Get one student's full profile by their ID."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        # 404 means "not found" — standard HTTP status code
        raise HTTPException(status_code=404, detail="Student not found")

    return dict(row)


@app.get("/stats")
def get_stats():
    """
    Returns summary statistics for the dashboard header cards.
    Total students, dropout count, risk distribution, etc.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students WHERE dropped_out = 1")
    total_dropouts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'high'")
    high_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'medium'")
    medium_risk = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'low'")
    low_risk = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(attendance_rate) FROM students")
    avg_attendance = round(cursor.fetchone()[0], 1)

    conn.close()

    return {
        "total_students": total,
        "total_dropouts": total_dropouts,
        "dropout_rate": round(total_dropouts / total * 100, 1),
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
        "avg_attendance": avg_attendance
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentInput):
    """
    Predict dropout risk for a NEW student not in the database.
    Accepts student data, returns risk score and top risk factors.
    """
    from sklearn.preprocessing import LabelEncoder

    # Encode gender the same way we did in model.py
    gender_encoded = 0 if student.gender == "M" else 1

    # Build input in the exact same feature order the model was trained on
    input_data = pd.DataFrame([{
        "attendance_rate": student.attendance_rate,
        "avg_grade": student.avg_grade,
        "absences_last_month": student.absences_last_month,
        "gender_encoded": gender_encoded,
        "distance_km": student.distance_km,
        "num_siblings": student.num_siblings,
        "fee_arrears": student.fee_arrears
    }])[feature_names]  # reorder columns to match training order

    # Get risk probability
    risk_score = float(model.predict_proba(input_data)[0][1])

    # Classify risk level
    if risk_score >= 0.6:
        risk_level = "high"
    elif risk_score >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Identify top risk factors to explain the prediction
    # Compare student values against safe thresholds
    risk_factors = []
    if student.attendance_rate < 70:
        risk_factors.append(f"Low attendance ({student.attendance_rate}%)")
    if student.avg_grade < 50:
        risk_factors.append(f"Low average grade ({student.avg_grade})")
    if student.fee_arrears > 5000:
        risk_factors.append(f"High fee arrears (KES {student.fee_arrears})")
    if student.distance_km > 15:
        risk_factors.append(f"Long distance to school ({student.distance_km}km)")
    if student.absences_last_month > 8:
        risk_factors.append(f"Many recent absences ({student.absences_last_month})")

    if not risk_factors:
        risk_factors = ["No major risk factors detected"]

    return PredictionResponse(
        name=student.name,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        top_risk_factors=risk_factors
    )