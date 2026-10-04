"""Pre-fetch the only model weights the pipeline uses (Depth Anything V2 Metric-Indoor Small,
~100 MB, Apache-2.0) into the Hugging Face cache, so the walk-in run needs no network."""
from transformers import AutoModelForDepthEstimation

from propscan.models.depth import MODEL_ID

if __name__ == "__main__":
    AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    print("ok:", MODEL_ID)
