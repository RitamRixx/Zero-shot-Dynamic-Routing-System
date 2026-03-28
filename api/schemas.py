# from pydantic import BaseModel, field_validator


# class PredictRequest(BaseModel):
#     text = str

#     @field_validator("text")
#     @classmethod
#     def text_must_not_be_empty(cls,v):
#         if not v.strip():
#             raise ValueError("text must not be empty or whitespace")
#         return v.strip()
    

# class UpdateLabelsRequest(BaseModel):
#     labels: list[str]

#     @field_validator("labels")
#     @classmethod
#     def labels_must_not_be_empty(cls,v):
#         if not v:
#             raise ValueError("labels list must not be empty")
#         return [label.strip() for label in v if label.strip()]
    
# class PredictResponse(BaseModel):
#     text: str
#     intent: str
#     confidence: float
#     is_oos: bool


# class LabelsResponse(BaseModel):
#     count: int
#     labels: list[str]


# class UpdateLabelsResponse(BaseModel):
#     previous_count: int
#     new_count: int
#     labels: list[str]

from pydantic import BaseModel, field_validator

# --- Request Models ---

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v.strip()


class UpdateLabelsRequest(BaseModel):
    labels: list[str]

    @field_validator("labels")
    @classmethod
    def labels_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("labels list must not be empty")
        return [label.strip() for label in v if label.strip()]


# --- Response Models ---

class PredictResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    is_oos: bool


class LabelsResponse(BaseModel):
    count: int
    labels: list[str]


class UpdateLabelsResponse(BaseModel):
    previous_count: int
    new_count: int
    labels: list[str]