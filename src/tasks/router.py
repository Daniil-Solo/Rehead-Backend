from fastapi import APIRouter, Query, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from tasks.dependencies import get_task
from tasks.celery_tasks import generate_content_task, celery
from tasks.schemas import CreateTask, TaskStatus, TaskResult, TaskInfo
from tasks.models import Task, TaskGeneratedText, TaskGeneratedImage
from database import get_async_session
from tasks.utils import from_bytes_to_base64

task_router = APIRouter(prefix="/tasks")


@task_router.post("/generate_content", response_model=TaskInfo)
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
    celery_task = generate_content_task.delay(task.id)
    return TaskInfo(celery_task_id=celery_task.id, db_task_id=task.id)


@task_router.get("/check_status", response_model=TaskStatus)
async def check_status(task_id: str = Query()):
    result = celery.AsyncResult(task_id)
    return TaskStatus(status=result.status)


@task_router.get("/get_result", response_model=TaskResult)
async def get_result(result_task_id: str = Query(), session: AsyncSession = Depends(get_async_session)):
    query = select(TaskGeneratedText).where(TaskGeneratedText.task_id == result_task_id)
    result = await session.execute(query)
    generated_text: TaskGeneratedText = result.scalar()
    query2 = select(TaskGeneratedImage).where(TaskGeneratedImage.task_id == result_task_id)
    result2 = await session.execute(query2)
    generated_images = result2.all()
    image_strings = [
        from_bytes_to_base64(generated_image.image, generated_image.filename) for generated_image in generated_images
    ]
    return TaskResult(text=generated_text.text, images=image_strings)
