from fastapi import FastAPI
from api.router import router

app = FastAPI(
    title="Zero-Shot Dynamic Routing System",
    description="Intent classification with runtime label updates — no retraining required.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {
        "project": "Zero-Shot Dynamic Routing System",
        "version": "1.0.0",
        "docs": "http://127.0.0.1:8000/docs"
    }

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}