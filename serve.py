import os
import argparse
import random
import sys
import uvicorn
import logging
import hashlib

# os.environ["ATTN_BACKEND"] = (
#     "xformers"  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
# )
# os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
# pipeline = TrellisImageTo3DPipeline.from_pretrained("models/TRELLIS-image-large")
pipeline.cuda()

logger = logging.getLogger("uvicorn.info")


class GenerationSettings(BaseModel):
    """Settings for the model inference"""

    # 0 means random seed
    seed: int = 0

    sparse_structure_sampler_steps: int = 12
    sparse_structure_sampler_cfg_strength: float = 7.5

    def get_seed(self) -> int:
        if self.seed == 0:
            return random.randint(0, sys.maxsize)
        else:
            return self.seed

    def get_hash(self):
        # Convert the object to a json dictionary
        obj_json = self.model_dump_json()

        # Generate SHA256 hash
        return hashlib.sha256(obj_json.encode("utf-8")).hexdigest()


def cache_filename(settings: GenerationSettings, image: Image) -> str:
    """Generate a cache filename for the generated asset
    out of combined hashes of the image and the settings
    :return: Cache filename for the given parameters
    """
    img_bytes = image.tobytes()
    img_hash = hashlib.sha256(img_bytes).hexdigest()
    settings_hash = settings.get_hash()
    return os.path.join(".", ".cache", f"{img_hash}_{settings_hash}.bin")


def get_from_cache(settings: GenerationSettings, image: Image) -> BytesIO | None:
    """Cache lookup for asset generation based on settings and image data

    :return: either the cached asset bytes or None if not found
    """
    filename = cache_filename(settings, image)

    if os.path.exists(filename) and os.path.isfile(filename):
        with open(filename, "rb") as fh:
            logger.info(f"Found {filename} in cache")
            return BytesIO(fh.read())
    else:
        return None


def put_into_cache(settings: GenerationSettings, image: Image, result: BytesIO):
    filename = cache_filename(settings, image)
    logger.info(f"Caching result image {filename}")

    cache_dir_name = os.path.join(".", ".cache")
    if not os.path.isdir(cache_dir_name):
        os.mkdir(cache_dir_name)

    with open(filename, "wb") as file:
        return file.write(result.getvalue())


app = FastAPI()


@app.get("/")
def read_root():
    return {"Welcome": "Trellis 3D asset generation"}


@app.post("/asset-from-image/")
def asset_from_image(
    image_file: UploadFile = File(...), settings: GenerationSettings = Depends()
):
    """Upload an image and create a 3D glb asset from it.

    Args:
         image_file (UploadFile): An image file of size 1048x1048. Plain background or Alpha
         settings (GenerationSettings): How to infer the model

    Returns:
         StreamingResponse: The processed image buffer in PNG format.
    """

    # Read the image file from the request
    image = Image.open(BytesIO(image_file.file.read()))

    # See if we have it in cache and if so, return it immediately
    cached_result = get_from_cache(settings, image)
    if cached_result:
        logger.info(f"Returning image from cache")
        cached_result.seek(0)
        # Return the processed image buffer
        return StreamingResponse(cached_result, media_type="model/gltf-binary")
    else:
        logger.info(f"Cache miss. Have to generate the image")

    seed = settings.get_seed()

    # Run the pipeline
    outputs = pipeline.run(
        image,
        # Optional parameters
        seed=seed,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )

    buffer = BytesIO()
    glb.export(file_obj=buffer, file_type="glb")
    buffer.seek(0)

    # Cache the image so we don't have to generate it again
    put_into_cache(settings, image, buffer)
    buffer.seek(0)

    # Return the processed image buffer
    return StreamingResponse(buffer, media_type="model/gltf-binary")


@app.post("/asset-from-storage/")
def asset_from_storage(
    image_file: UploadFile = File(...), settings: GenerationSettings = Depends()
):
    """Get a previously generated 3D glb asset from storage.
    This is mostly for testing of the consumer so it doesn't need to generate every time.

    Args:
         image_file (UploadFile): An image file of size 1048x1048. Plain background or Alpha
         settings (GenerationSettings): How to infer the model

    Returns:
         StreamingResponse: The processed image buffer in PNG format.
    """

    # Read the image file from the request
    image = Image.open(BytesIO(image_file.file.read()))
    logger.info(f"read image of size {image.size}")

    seed = settings.get_seed()
    logger.info(f"seed is {seed}")

    with open("C:\\SAPDevelop\\trellis\\sample.glb", "rb") as fh:
        buffer = BytesIO(fh.read())

    # Return the processed image buffer
    return StreamingResponse(buffer, media_type="model/gltf-binary")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="expose API on this port"
    )
    args = parser.parse_args()

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
        log_level="info",
        workers=1,
    )
