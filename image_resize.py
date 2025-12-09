import io
import os
from PIL import Image

MAX_FILE_SIZE_MB = 10.0
MAX_RESOLUTION = 1024


def load_image_with_auto_resize(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    img = Image.open(path).convert("RGB")

    if file_size_mb <= MAX_FILE_SIZE_MB:
        return img

    print(
        f"[INFO] Image {path} is too large ({file_size_mb:.2f} MB) "
        f"=> resizing (max {MAX_RESOLUTION}px)..."
    )

    img.thumbnail((MAX_RESOLUTION, MAX_RESOLUTION))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    buffer.seek(0)

    resized_img = Image.open(buffer).convert("RGB")
    return resized_img
