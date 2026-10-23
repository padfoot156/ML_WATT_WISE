"""
WattWise AI - FastAPI Application Server
Serves RESTful analytical endpoints, figure assets, and interactive web dashboard.
"""

import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import router

app = FastAPI(
    title="WattWise AI — Energy Demand, Supply & Expansion Intelligence",
    description="Full-stack analytical and forecasting engine for Grid-India electricity demand and renewable expansion.",
    version="1.0.0"
)

# Configurable Production CORS
cors_env = os.getenv("CORS_ORIGINS", "*")
origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health check endpoint for cloud platform probes (Render, AWS, GCP, etc.)
@app.get("/health", tags=["Health"])
def root_health():
    """Root health check for load balancer and cloud provider monitoring."""
    return {
        "status": "healthy",
        "service": "WattWise AI Analytical Engine",
        "version": "1.0.0"
    }

# Mount API endpoints
app.include_router(router, prefix="/api")

# Mount figures directory for direct image serving
figures_dir = os.path.abspath("results/figures")
os.makedirs(figures_dir, exist_ok=True)
app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the WattWise AI unified interactive dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>Dashboard HTML not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)
