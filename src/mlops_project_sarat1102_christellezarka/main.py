from fastapi import FastAPI

from mlops_project_sarat1102_christellezarka.endpoints.health import router as health_router
from mlops_project_sarat1102_christellezarka.endpoints.pipeline import router as pipeline_router

from prometheus_fastapi_instrumentator import Instrumentator
app = FastAPI(title="ML Data Pipeline API", version="1.0")
Instrumentator().instrument(app).expose(app)
# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

