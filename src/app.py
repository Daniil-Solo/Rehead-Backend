from fastapi import FastAPI, Depends

from auth.models import User
from auth.router import auth_router, register_router, current_active_user
from uvicorn import run

app = FastAPI()

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
