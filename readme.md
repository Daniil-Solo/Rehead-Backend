## Пример .env
```cmd
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=your_db
DB_USER=your_user
DB_PASS=your_pass
SECRET_KEY=your_secret_key
REDIS_URL=redis://127.0.0.1:6379
NLP_MODEL_PATH=/your/path
ALLOWED_HOSTS=your_host
```

# Запуск
Миграции
```cmd
alembic upgrade ec0e168fe34a
```
Веб-сервис
```cmd
gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:5000
```
Celery-worker
```cmd
celery -A tasks.celery_tasks worker
```
Celery Flower
```cmd
celery -A celery_tasks.py flower --port=5555
```