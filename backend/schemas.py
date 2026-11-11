from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class HealthResponse(BaseModel):
    status: str
    service: str


class APIInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    endpoints: List[str]


class UploadResponse(BaseModel):
    success: bool
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    missing_values: int
    preview: List[Dict[str, Any]]


class GenerateResponse(BaseModel):
    success: bool
    result_id: str
    generator: str
    message: str


class ResultResponse(BaseModel):
    success: bool
    result_id: str
    rows: int
    columns: int
    generator: str
    configuration: Dict[str, Any]
    validation: Dict[str, Any]
    metrics: Dict[str, Any]
    preview: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
