from uuid import UUID
from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    celery_task_id: UUID
    db_task_id: UUID


class TaskStatus(BaseModel):
    status: str = "PENDING"


class TaskResult(BaseModel):
    text: str
    images: list[str]


class CreateTask(BaseModel):
    text: str
    remove_background: bool
    generate_background: bool
