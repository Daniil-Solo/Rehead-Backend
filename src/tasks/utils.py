import base64


def from_bytes_to_base64(image_bytes: bytes, filename: str = "image.png"):
    format = filename.rsplit('.', 1)[1]
    prefix = f'data:image/{format};base64,'.encode()
    base64_string = base64.b64encode(image_bytes)
    return prefix + base64_string
