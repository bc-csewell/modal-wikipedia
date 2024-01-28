from modal import Image, Secret, Stub, Volume, build, enter, exit, gpu, method

# We first set out configuration variables for our script.
## Embedding Containers Configuration
GPU_CONCURRENCY = 100
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]
BATCH_SIZE = 512
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

## Dataset-Specific Configuration
DATASET_NAME = "wikipedia"
DATASET_READ_VOLUME = Volume.persisted("embedding-wikipedia")
EMBEDDING_CHECKPOINT_VOLUME = Volume.persisted("checkpoint")
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = True

## Upload-Specific Configuration
DATASET_HF_UPLOAD_REPO_NAME = "567-labs/upload-test"
UPLOAD_TO_HF = True

def upload_result_to_hf(batch_size: int) -> None:
    """
    Uploads the result to the Hugging Face Hub.

    Args:
        batch_size (int): The batch size for the model.

    Returns:
        None
    """
    import os
    import time

    from huggingface_hub import HfApi

    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}"
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Pushing to hub {DATASET_HF_UPLOAD_REPO_NAME}")
    start = time.perf_counter()
    api.upload_folder(
        folder_path=path_parent_folder,
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    end = time.perf_counter()
    print(f"Uploaded in {end-start}s")