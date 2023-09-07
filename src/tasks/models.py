import uuid
from sqlalchemy import Boolean, String, ForeignKey
from sqlalchemy.types import Uuid, LargeBinary
from sqlalchemy.orm import mapped_column, Mapped
from src.database import Base


class Task(Base):
    __tablename__ = "task"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    # user_id: Mapped[uuid.UUID] = mapped_column(
    #     ForeignKey("user.id"), nullable=False
    # )
    text: Mapped[str] = mapped_column(
        String, nullable=False
    )
    remove_background: Mapped[bool] = mapped_column(
        Boolean, nullable=False
    )
    generate_background: Mapped[bool] = mapped_column(
        Boolean, nullable=False
    )
    filename: Mapped[str] = mapped_column(
        String, nullable=False
    )
    image: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )


class TaskGeneratedText(Base):
    __tablename__ = "task_generated_text"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task.id"), nullable=False
    )
    text: Mapped[str] = mapped_column(
        String, nullable=False
    )


class TaskGeneratedImage(Base):
    __tablename__ = "task_generated_image"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task.id"), nullable=False
    )
    filename: Mapped[str] = mapped_column(
        String, nullable=False
    )
    image: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False
    )
