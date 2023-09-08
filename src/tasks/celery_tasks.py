import uuid
from sqlalchemy import select
from celery import Celery
from config import REDIS_URL, NLP_MODEL_PATH
from tasks.ai_models.DescGenModel import DescGenModel
from tasks.ai_models.BackGenModel import remove_background, BackGenModel
from tasks.models import Task, TaskGeneratedText, TaskGeneratedImage
from database import Session


celery = Celery("tasks", broker=REDIS_URL)


@celery.task
def generate_content_task(task_id: uuid.UUID):
    session = Session()
    query = select(Task).where(Task.id == task_id)
    task: Task = session.execute(query).scalar()
    generated_text = DescGenModel(NLP_MODEL_PATH).infer(task.text)
    task_generated_text = TaskGeneratedText(task_id=task_id, text=generated_text)
    task_generated_images = [TaskGeneratedImage(image_bytes=task.image, filename=task.filename, task_id=task_id)]
    if task.remove_background:
        tgi = TaskGeneratedImage(image_bytes=remove_background(task.image), filename="image.png", task_id=task_id)
        task_generated_images.append(tgi)
    if task.generate_background:
        for image_bytes in BackGenModel().infer(task.image):
            tgi = TaskGeneratedImage(image_bytes=image_bytes, filename="image.png", task_id=task_id)
            task_generated_images.append(tgi)
    session.add_all(task_generated_images)
    session.add(task_generated_text)
    session.commit()
    session.close()
    return task_id
