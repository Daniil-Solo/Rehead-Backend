from fastapi import APIRouter, Query, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from tasks.dependencies import get_task
from tasks.celery_tasks import celery, generate_product_description, generate_product_images
from tasks.schemas import CreateTask, TaskStatus, TaskInfo, TaskDescriptionResult, TaskImagesResult
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
    celery_image_task = generate_product_images.delay(task.id)
    celery_description_task = generate_product_description.delay(task.id)
    return TaskInfo(
        celery_image_task_id=celery_image_task.id,
        celery_description_task_id=celery_description_task.id,
        db_task_id=task.id
    )


@task_router.get("/check_status", response_model=TaskStatus)
async def check_status(task_id: str = Query()):
    result = celery.AsyncResult(task_id)
    return TaskStatus(status=result.status)


@task_router.get("/get_description_result", response_model=TaskDescriptionResult)
async def get_description_result(result_task_id: str = Query(), session: AsyncSession = Depends(get_async_session)):
    query = select(TaskGeneratedText).where(TaskGeneratedText.task_id == result_task_id)
    result = await session.execute(query)
    generated_text: TaskGeneratedText = result.scalar()
    return TaskDescriptionResult(text=generated_text.text)


@task_router.get("/get_images_result", response_model=TaskImagesResult)
async def get_images_result(result_task_id: str = Query(), session: AsyncSession = Depends(get_async_session)):
    query = select(TaskGeneratedImage).where(TaskGeneratedImage.task_id == result_task_id)
    result = await session.execute(query)
    generated_images = result.scalars().all()
    image_strings = [
        from_bytes_to_base64(generated_image.image, generated_image.filename) for generated_image in generated_images
    ]
    return TaskImagesResult(images=image_strings)
