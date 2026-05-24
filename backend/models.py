# models.py
# ---------
# Pydantic models define the SHAPE of data in our API.
# They serve two purposes:
# 1. VALIDATION — if incoming data doesn't match the shape, FastAPI
#    automatically rejects it with a helpful error message.
# 2. DOCUMENTATION — FastAPI reads these to generate the /docs page.
#
# Think of them as contracts:
# "I promise this data will always have exactly these fields."

from pydantic import BaseModel, Field
from typing import Optional

# ------------------------------------------------------------------
# What a Student record looks like when returned from the database
# ------------------------------------------------------------------
class Student(BaseModel):
    """Represents a full student record from the database."""
    id: int
    name: str
    gender: str
    attendance_rate: float
    avg_grade: float
    absences_last_month: int
    distance_km: float
    num_siblings: int
    fee_arrears: int
    dropped_out: int
    risk_score: float
    risk_level: str  # "low", "medium", or "high"

# ------------------------------------------------------------------
# What data we need to predict risk for a NEW student
# ------------------------------------------------------------------
class StudentInput(BaseModel):
    """
    Data required to predict risk for a new student.
    Field(...) means the field is REQUIRED.
    ge = greater than or equal to
    le = less than or equal to
    These constraints are automatically enforced by FastAPI.
    """
    name: str
    gender: str = Field(..., pattern="^[MF]$",
                        description="M for Male, F for Female")
    attendance_rate: float = Field(..., ge=0, le=100,
                                   description="Attendance percentage")
    avg_grade: float = Field(..., ge=0, le=100,
                             description="Average grade (0-100)")
    absences_last_month: int = Field(..., ge=0,
                                      description="Absences in last month")
    distance_km: float = Field(..., ge=0,
                               description="Distance to school in km")
    num_siblings: int = Field(..., ge=0,
                              description="Number of siblings")
    fee_arrears: int = Field(..., ge=0,
                             description="Unpaid fees in KES")

# ------------------------------------------------------------------
# What a prediction response looks like when returned to frontend
# ------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Data returned after predicting risk for a student."""
    name: str
    risk_score: float        # 0.0 to 1.0
    risk_level: str          # "low", "medium", "high"
    top_risk_factors: list[str]  # human-readable explanation

# ------------------------------------------------------------------
# What the stats endpoint returns
# ------------------------------------------------------------------
class StatsResponse(BaseModel):
    """Summary statistics for the dashboard header cards."""
    total_students: int
    total_dropouts: int
    dropout_rate: float
    high_risk: int
    medium_risk: int
    low_risk: int
    avg_attendance: float