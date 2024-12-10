from pathlib import Path
from PIL import Image
from datetime import datetime
import numpy as np

import io
import base64

from typing import Union, List, Tuple


def bytes_to_image_pil(raw_image: bytes) -> Image:
    img_ary = np.frombuffer(raw_image, np.uint8)
    return Image.open(io.BytesIO(img_ary))


def image_pil_to_buffer(image: Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_to_base64(image: Image.Image, quality: int) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(string: str) -> Image:
    msg = base64.b64decode(string)
    buf = io.BytesIO(msg)
    return Image.open(buf)


# ----- image manipulation
def resize_image(img: Image.Image, max_length: int | float) -> Image.Image:
    """Resize an image to a maximum width while maintaining its aspect ratio."""
    # Calculate the new width and height while maintaining the aspect ratio
    width, height = img.size  # PIL: (width, height)

    len_max = max(img.size)
    len_min = min(img.size)

    if len_max > max_length:
        len_max_new = max_length
        len_min_new = int(len_min * (max_length / len_max))
    else:
        len_max_new = len_max
        len_min_new = len_min

    size_new = (len_max_new, len_min_new) if width > height else (len_min_new, len_max_new)

    # return the resized image
    return img.resize(size_new, Image.BICUBIC)

def save_image(
        img: Image,
        image_extension: str,
        folder: Union[str, Path] = None,
        note: Union[str, List[str]] = None,
        quality: int = 90
) -> Union[Path, None]:
    if note is None:
        notes = []
    elif isinstance(note, str):
        notes = [""] + [note]
    elif isinstance(note, list):
        notes = [""] + [f"{el}" for el in note]
    else:
        raise TypeError(f"Expecting input 'note' to be a string or a list of strings but was {type(note)}.")

    # create filename from current timestamp
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_".join(notes)
    # create full path (not necessarily absolute)
    folder = Path(folder)
    if not folder.is_dir():
        folder.mkdir(exist_ok=True)
    path_to_file = Path(folder) / filename

    # file extension
    extension = image_extension.strip('.').lower()
    if extension == "jpeg":
        extension = "jpg"

    # save image
    if not path_to_file.exists():
        img.save(path_to_file.with_suffix(f".{extension}"), quality=quality)
        return path_to_file
    else:
        return None
