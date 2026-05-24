# main.py
# -------
# This is the entry point of the entire backend.
# Its only job is to:
# 1. Create the FastAPI app
# 2. Configure CORS
# 3. Connect the routers
# 4. Initialize the database on startup
#
# Notice how short this file is compared to before.
# All the real logic lives in its own dedicated file.
# This is called "separation of concerns" — a core principle
# of professional software development.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import our routers — each one brings its own endpoints
from routers import students, stats

# Import database initializer and the ML model
import database
import predict

# ------------------------------------------------------------------
# STEP 1: Create the FastAPI application instance
# ------------------------------------------------------------------
app = FastAPI(
    title="BridgeClass API",
    description="AI-powered student dropout early-warning system for Kenyan schools",
    version="2.0.0"
)

# ------------------------------------------------------------------
# STEP 2: Configure CORS
# ------------------------------------------------------------------
# CORS must be added BEFORE registering routes.
# Order matters in FastAPI middleware — this is a common gotcha.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Create React App default port
        "http://localhost:5173",   # Vite default port (our frontend)
    ],
    allow_credentials=True,
    allow_methods=["*"],     # allow GET, POST, PUT, DELETE etc.
    allow_headers=["*"],     # allow all headers
)

# ------------------------------------------------------------------
# STEP 3: Register routers
# ------------------------------------------------------------------
# include_router plugs each router into the main app.
# From this moment, all routes defined in those files are active.

app.include_router(students.router)
app.include_router(stats.router)

# ------------------------------------------------------------------
# STEP 4: Initialize database on startup
# ------------------------------------------------------------------
# This runs once when the server starts.
# We pass the model and feature names from predict.py
# so the database can compute risk scores during seeding.

@app.on_event("startup")
async def startup_event():
    """
    FastAPI calls this function automatically when the server starts.
    Think of it as the "opening checklist" before the restaurant opens.
    """
    print("\n🚀 BridgeClass API starting up...")
    database.init_db(
        model=predict.get_model(),
        feature_names=predict.get_feature_names()
    )
    print("✅ BridgeClass API is ready\n")

# ------------------------------------------------------------------
# STEP 5: Root endpoint
# ------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """
    Health check endpoint.
    A simple way to confirm the API is running.
    Monitoring tools ping this endpoint to check server health.
    """
    return {
        "status": "running",
        "message": "BridgeClass API is live",
        "version": "2.0.0",
        "docs": "/docs"
    }