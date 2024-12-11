import os
import random
import sys
import uvicorn
import logging

os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
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
# pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = TrellisImageTo3DPipeline.from_pretrained("models/TRELLIS-image-large")
pipeline.cuda()


class GenerationSettings(BaseModel):
    """Settings for the model inference
    """
    # 0 means random seed
    seed: int = 0

    sparse_structure_sampler_steps: int = 12
    sparse_structure_sampler_cfg_strength: float = 7.5

    def get_seed(self) -> int:
        if self.seed == 0:
            return random.randint(0, sys.maxsize)
        else:
            return self.seed


app = FastAPI()
logger = logging.getLogger('uvicorn.info')


@app.get("/")
def read_root():
    return {"Welcome": "Trellis 3D asset generation"}


@app.post("/asset-from-image/")
def asset_from_image(image_file: UploadFile = File(...), settings: GenerationSettings = Depends()):
    """Upload an image and create a 3D glb asset from it.

    Args:
         image_file (UploadFile): An image file of size 1048x1048. Plain background or Alpha
         settings (GenerationSettings): How to infer the model

    Returns:
         StreamingResponse: The processed image buffer in PNG format.
    """

    # Read the image file from the request
    image = Image.open(BytesIO(image_file.file.read()))

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
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )

    buffer = BytesIO()
    glb.export(file_obj=buffer, file_type="glb")
    buffer.seek(0)

    # Return the processed image buffer
    return StreamingResponse(buffer, media_type="model/gltf-binary")


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False, log_level="info", workers=1)

