from uuid import UUID
from pydantic import BaseModel, Field


class TaskStatus(BaseModel):
    task_id: UUID
    status: str = "PENDING"


class TaskResult(BaseModel):
    text: str
    images: list[str]


class CreateTask(BaseModel):
    text: str
    remove_background: bool
    generate_background: bool
