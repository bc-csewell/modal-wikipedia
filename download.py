from modal import Image, Volume, Stub
from config import DATASET_READ_VOLUME, DATASET_DIR

image = Image.debian_slim().pip_install("datasets")

stub = Stub(image=image)

@stub.function(volumes={DATASET_DIR: DATASET_READ_VOLUME}, timeout=3000)
def download_dataset(cache=False):
    from datasets import load_dataset

    # Download and save the dataset locally
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=10)
    dataset.save_to_disk(f"{DATASET_DIR}/wikipedia")

    # Commit and save to the volume
    DATASET_READ_VOLUME.commit()