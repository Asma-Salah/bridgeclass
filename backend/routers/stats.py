# routers/stats.py
# ----------------
# Handles all endpoints related to summary statistics.
# These feed the dashboard header cards in the frontend.
#
# Endpoints in this file:
#   GET /stats → dashboard summary numbers

from fastapi import APIRouter
from models import StatsResponse
import database

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"]
)

# ------------------------------------------------------------------
# GET /stats
# ------------------------------------------------------------------
@router.get("/", response_model=StatsResponse)
def get_stats():
    """
    Returns summary statistics for the dashboard.

    This is what powers the header cards the teacher sees:
    - Total students enrolled
    - Total who dropped out
    - Overall dropout rate
    - Count of high / medium / low risk students
    - Average attendance across all students
    """
    return database.get_stats()