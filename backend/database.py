# database.py
# -----------
# This file handles everything SQLite-related.
# One rule: NO prediction logic, NO API logic here.
# This file only knows about the database.
#
# Why separate this from main.py?
# If we ever swap SQLite for PostgreSQL, we only change THIS file.
# Nothing else in the project needs to change.

import sqlite3
import pandas as pd

# The database file path — SQLite creates this file automatically
DB_PATH = "bridgeclass.db"

# ------------------------------------------------------------------
# STEP 1: Connection helper
# ------------------------------------------------------------------
def get_db():
    """
    Creates and returns a database connection.

    conn.row_factory = sqlite3.Row is important —
    without it, SQLite returns data as plain tuples: (1, "Amina", 82.5...)
    with it, SQLite returns data as dictionaries: {"id": 1, "name": "Amina"...}
    Dictionaries are much easier to work with in Python and JSON.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------------------------------------------
# STEP 2: Initialize the database
# ------------------------------------------------------------------
def init_db(model, feature_names: list):
    """
    Creates the students table and seeds it with data from CSV.
    Called ONCE when the server starts.

    Args:
        model: the trained Random Forest model (used to compute risk scores)
        feature_names: list of feature column names in the correct order
    """
    conn = get_db()
    cursor = conn.cursor()

    # CREATE TABLE IF NOT EXISTS means:
    # "create this table, but if it already exists, do nothing"
    # This makes the function safe to call multiple times
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL,
            gender              TEXT,
            attendance_rate     REAL,
            avg_grade           REAL,
            absences_last_month INTEGER,
            distance_km         REAL,
            num_siblings        INTEGER,
            fee_arrears         INTEGER,
            dropped_out         INTEGER,
            risk_score          REAL,
            risk_level          TEXT
        )
    """)

    # Check if table already has data
    # If it does, skip seeding — we don't want duplicates
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]

    if count == 0:
        print("⏳ Seeding database from CSV...")
        _seed_students(cursor, model, feature_names)
        conn.commit()
        print(f"✅ Database seeded successfully")
    else:
        print(f"✅ Database already contains {count} students — skipping seed")

    conn.close()

# ------------------------------------------------------------------
# STEP 3: Private seeding function
# ------------------------------------------------------------------
def _seed_students(cursor, model, feature_names: list):
    """
    Reads students.csv, computes risk scores, inserts all rows.

    The underscore prefix (_seed_students) is a Python convention
    meaning "this function is private — only call it from inside
    this file, not from outside."
    """
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("data/students.csv")

    # Encode gender: M=0, F=1
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(df["gender"])

    # Get risk scores from the ML model
    X = df[feature_names]
    risk_scores = model.predict_proba(X)[:, 1]
    df["risk_score"] = risk_scores.round(4)
    df["risk_level"] = df["risk_score"].apply(_classify_risk)

    # Insert every student into the database
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

# ------------------------------------------------------------------
# STEP 4: Query functions
# ------------------------------------------------------------------
def get_all_students(risk_level: str = None, limit: int = 100):
    """
    Fetch students from database, highest risk first.
    Optionally filter by risk_level: "high", "medium", or "low"
    """
    conn = get_db()
    cursor = conn.cursor()

    if risk_level:
        cursor.execute("""
            SELECT * FROM students
            WHERE risk_level = ?
            ORDER BY risk_score DESC
            LIMIT ?
        """, (risk_level, limit))
    else:
        cursor.execute("""
            SELECT * FROM students
            ORDER BY risk_score DESC
            LIMIT ?
        """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    # dict(row) converts SQLite.Row → regular Python dictionary
    return [dict(row) for row in rows]


def get_student_by_id(student_id: int):
    """
    Fetch a single student by their ID.
    Returns None if not found — the router handles the 404 response.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return dict(row)


def get_stats():
    """
    Returns summary statistics for the dashboard.
    Uses SQL aggregation functions — COUNT, AVG.
    """
    conn = get_db()
    cursor = conn.cursor()

    # SQL COUNT(*) counts all rows matching the condition
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

    # AVG is a SQL aggregation function — like pandas .mean()
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

# ------------------------------------------------------------------
# STEP 5: Helper function
# ------------------------------------------------------------------
def _classify_risk(score: float) -> str:
    """
    Converts a raw probability score (0.0-1.0) into a risk label.
    These thresholds were chosen based on the score distribution.

    >= 0.6 → high   (top concern, needs immediate attention)
    >= 0.3 → medium (worth monitoring)
    <  0.3 → low    (currently safe)
    """
    if score >= 0.6:
        return "high"
    elif score >= 0.3:
        return "medium"
    else:
        return "low"