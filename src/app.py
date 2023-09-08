from uvicorn import run
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth.router import auth_router, register_router
from config import ALLOWED_HOSTS
from tasks.router import task_router

app = FastAPI()

origins = ALLOWED_HOSTS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    register_router,
    prefix="/users",
    tags=["users"]
)
app.include_router(
    auth_router,
    prefix="/users/auth",
    tags=["users"]
)
app.include_router(
    task_router,
    tags=["tasks"]
)


if __name__ == "__main__":
    run(app, port=5000)