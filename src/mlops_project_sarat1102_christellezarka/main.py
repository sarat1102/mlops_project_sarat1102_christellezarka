from fastapi import FastAPI

from ml_data_pipeline.endpoints.health import router as health_router
from ml_data_pipeline.endpoints.pipeline import router as pipeline_router

app = FastAPI(title="ML Data Pipeline API", version="1.0")

# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

