from fastapi_users.authentication import BearerTransport, JWTStrategy, AuthenticationBackend
from config import SECRET_KEY


cookie_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET_KEY, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="rehead_auth",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)
