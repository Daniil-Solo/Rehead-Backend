import time

from celery import Celery
from config import REDIS_URL
celery = Celery("tasks", broker=REDIS_URL)


@celery.task
def test_predict(seconds: int):
    time.sleep(seconds)
    return "OK"
