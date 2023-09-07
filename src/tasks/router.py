import base64
import random
import uuid
from typing import Annotated

from fastapi import APIRouter, Query, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from tasks.dependencies import get_task
from tasks.celery_tasks import test_predict
from tasks.schemas import CreateTask, TaskStatus, TaskResult
from tasks.models import Task, TaskGeneratedText, TaskGeneratedImage
from database import get_async_session

task_router = APIRouter(prefix="/tasks")


@task_router.get("/sleep")
async def predict_endpoint(second: int = Query(default=3, lt=10)):
    task = test_predict.delay(second)
    print(task)
    return {"status": str(task)}


@task_router.post("/generate_content", response_model=TaskStatus)
async def generate_content(new_task: CreateTask = Depends(get_task), file: UploadFile = File(...),
                           session: AsyncSession = Depends(get_async_session)):
    task = Task(
        text=new_task.text, remove_background=new_task.remove_background,
        generate_background=new_task.generate_background,
        filename=file.filename, image=file.file.read()
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return TaskStatus(task_id=task.id)


@task_router.get("/check_status", response_model=TaskStatus)
def check_status(task_id: str = Query()):
    random_value = random.random()
    print(random_value)
    if random_value > 0.5:
        return TaskStatus(task_id=task_id, status="STARTED")
    else:
        return TaskStatus(task_id=task_id, status="SUCCESS")


@task_router.get("/get_result", response_model=TaskResult)
def get_result(result_task_id: str = Query()):
    with open("tasks/288619.png", "rb") as image:
        image_bytes = image.read()
    image_str = base64.b64encode(image_bytes)
    return TaskResult(text="Результат генерации", images=[image_str])
