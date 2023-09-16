from uuid import UUID
from pydantic import BaseModel


class TaskInfo(BaseModel):
    celery_image_task_id: UUID
    celery_description_task_id: UUID
    db_task_id: UUID


class TaskStatus(BaseModel):
    status: str = "PENDING"


class TaskDescriptionResult(BaseModel):
    text: str


class TaskImagesResult(BaseModel):
    images: list[str]


class CreateTask(BaseModel):
    text: str
    remove_background: bool
    generate_background: bool
