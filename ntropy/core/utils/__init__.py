import tempfile
from PIL import Image
import requests
from ntropy.core.utils.base_format import Document
import os



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

def clear_cache():
    for file in temps_images:
        os.remove(file)
    print('cache cleared !')


