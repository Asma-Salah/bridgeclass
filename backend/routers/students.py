# routers/students.py
# -------------------
# Handles all endpoints related to students.
# This file only knows about students — nothing else.
#
# Endpoints in this file:
#   GET  /students          → all students (with optional filters)
#   GET  /students/{id}     → one student by ID
#   POST /students/predict  → predict risk for a new student

from fastapi import APIRouter, HTTPException, Query
from models import StudentInput, PredictionResponse
import database
import predict

# APIRouter works exactly like FastAPI app — but it is a piece, not the whole.
# prefix="/students" means every route here automatically starts with /students
# tags=["students"] groups them together in the /docs page
router = APIRouter(
    prefix="/students",
    tags=["Students"]
)

# ------------------------------------------------------------------
# GET /students
# ------------------------------------------------------------------
@router.get("/")
def get_all_students(
    risk_level: str = Query(
        default=None,
        description="Filter by risk level: high, medium, or low"
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Max number of students to return"
    )
):
    """
    Returns all students sorted by risk score (highest first).

    Optional query parameters:
    - ?risk_level=high    → only high risk students
    - ?risk_level=medium  → only medium risk students
    - ?limit=50           → return only 50 students

    Query parameters appear after ? in the URL.
    Example: /students?risk_level=high&limit=20
    """
    students = database.get_all_students(
        risk_level=risk_level,
        limit=limit
    )
    return students


# ------------------------------------------------------------------
# GET /students/{student_id}
# ------------------------------------------------------------------
@router.get("/{student_id}")
def get_student(student_id: int):
    """
    Returns one student's full profile by their ID.

    {student_id} is a PATH parameter — it is part of the URL itself.
    Example: /students/42 → returns student with id=42

    Path parameters vs Query parameters:
    - Path:  /students/42        → identifies a specific resource
    - Query: /students?limit=50  → filters or modifies the response
    """
    student = database.get_student_by_id(student_id)

    if student is None:
        # HTTPException with 404 tells the client "this resource doesn't exist"
        # This is standard HTTP protocol — every API follows this convention
        raise HTTPException(
            status_code=404,
            detail=f"Student with id {student_id} not found"
        )

    return student


# ------------------------------------------------------------------
# POST /students/predict
# ------------------------------------------------------------------
@router.post("/predict", response_model=PredictionResponse)
def predict_risk(student: StudentInput):
    """
    Predicts dropout risk for a NEW student not in the database.

    Why POST instead of GET?
    GET requests should only retrieve data — they have no body.
    POST requests send data to be processed — they have a body.
    We are sending student data to be processed by the model → POST.

    The response_model=PredictionResponse tells FastAPI:
    "validate and format the response using this Pydantic model"
    It also makes the /docs page show the expected response shape.
    """
    result = predict.predict_student_risk(
        attendance_rate=student.attendance_rate,
        avg_grade=student.avg_grade,
        absences_last_month=student.absences_last_month,
        gender=student.gender,
        distance_km=student.distance_km,
        num_siblings=student.num_siblings,
        fee_arrears=student.fee_arrears
    )

    return PredictionResponse(
        name=student.name,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        top_risk_factors=result["top_risk_factors"]
    )