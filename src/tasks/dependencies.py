from fastapi import Form
from tasks.schemas import CreateTask


def get_task(text: str = Form(), remove_background: bool = Form(), generate_background: bool = Form()) -> CreateTask:
    return CreateTask(
        text=text,
        remove_background=remove_background,
        generate_background=generate_background
    )
