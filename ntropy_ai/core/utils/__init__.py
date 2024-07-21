import tempfile
from PIL import Image
import requests
from ntropy_ai.core.utils.base_format import Document
import os
from shutil import get_terminal_size
from threading import Thread
import time
from itertools import cycle


temps_images = [] # we save the images in a list to be able to clear the cache
# save http img 
# by default return file path, if return_doc is True return file object
def save_img_to_temp_file(image_url: str, return_doc: bool = False):
    if image_url.startswith('http'):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as image_file:
            image_path = image_file.name
            img = Image.open(requests.get(image_url, stream=True).raw)
            # fix OSError: cannot write mode RGBA as JPEG
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(image_path)
            temps_images.append(image_path)
            if return_doc:
                return Document(image=image_path, page_number=-1, data_type="image")
            else:
                return image_path
    else:
        return image_url

def ensure_local_file(remote_file_path: str) -> str:
    if remote_file_path.startswith('http'):
        response = requests.get(remote_file_path)
        ext = os.path.splitext(remote_file_path)[1]  # Extract the file extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        return remote_file_path

def clear_cache():
    for file in temps_images:
        os.remove(file)
    print('cache cleared !')

    
def resize_image(image_path: str, max_size: int = 1024):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)  # Preserves aspect ratio
    new_image_path = os.path.splitext(image_path)[0] + ".png"
    img.save(new_image_path, "PNG")
    return new_image_path

class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
