from fastapi import APIRouter
from src.classifier import classifier
from api.schemas import (
    PredictRequest, PredictResponse,
    LabelsResponse,
    UpdateLabelsRequest, UpdateLabelsResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return classifier.classify(request.text)


@router.get("/labels", response_model=LabelsResponse)
def get_labels():
    labels = classifier.get_labels()
    return {"count": len(labels), "labels": labels}


@router.put("/update-labels", response_model=UpdateLabelsResponse)
def update_labels(request: UpdateLabelsRequest):
    return classifier.update_labels(request.labels)
