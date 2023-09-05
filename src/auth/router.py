import uuid

from fastapi_users import FastAPIUsers

from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from auth.settings import auth_backend
from auth.models import User

routers_manager = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

auth_router = routers_manager.get_auth_router(auth_backend)
register_router = routers_manager.get_register_router(UserRead, UserCreate)
current_active_user = routers_manager.current_user(active=True)
