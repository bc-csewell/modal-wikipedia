import asyncio
import subprocess
from config import DATASET_READ_VOLUME, DATASET_DIR, MODEL_ID, BATCH_SIZE, INTERMEDIATE_DIR, EMBEDDING_INTERMEDIATE_VOLUME, MODEL_SLUG

from modal import Image, gpu, build, enter, exit, method, Stub, Secret

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
]

tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx")
)

with tei_image.imports():
    import numpy as np

stub = Stub("example-embeddings")

def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")

@stub.cls(
    gpu= gpu.A10G(),
    image=tei_image, # This is defined above
    concurrency_limit=50,  # Number of concurrent containers that can be spawned to handle the task
)
class TextEmbeddingsInference:
    @build()
    def download_model(self):
        # Wait for server to start. This downloads the model weights when not present.
        spawn_server()

    @enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    @exit()
    def terminate_connection(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""

        # in order to send more data per request, we batch requests to
        # `TextEmbeddingsInference` and make concurrent requests to the endpoint
        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
        ]

        embeddings = np.concatenate(await asyncio.gather(*coros))
        return chunks, embeddings

def generate_chunks_from_dataset(xs, chunk_size: int = 400):
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )

def generate_batches(xs, batch_size=512):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

@stub.function(
    image=Image.debian_slim().pip_install("datasets", "pyarrow", "hf_transfer", "huggingface_hub"),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        INTERMEDIATE_DIR: EMBEDDING_INTERMEDIATE_VOLUME,
    },
    timeout=86400,
    secret=Secret.from_name("HF_TOKEN"),
)
def embed_dataset(batch_size: int = 512 * 50):
    from datasets import load_from_disk

    dataset = load_from_disk(f"{DATASET_DIR}/wikipedia")
    model = TextEmbeddingsInference()

    text_chunks = generate_chunks_from_dataset(dataset["train"], chunk_size=512)
    batches = generate_batches(text_chunks, batch_size=batch_size)

    # Collect the chunks and embeddings
    acc_chunks = []
    acc_embeddings  = []
    for batch_chunks, batch_embeddings in model.embed.map(batches, order_outputs=False):
        
        if isinstance(batch_chunks, batch_embeddings, Exception):
            print(f"Exception: {batch_chunks, batch_embeddings}")
            continue

        acc_chunks.extend(batch_chunks)
        acc_embeddings.extend(batch_embeddings)

    save_dataset_to_intermediate_volume(
        acc_chunks, acc_embeddings, batch_size
    )

    return

def save_dataset_to_intermediate_volume(acc_chunks, acc_embeddings, batch_size):
    """Saves the dataset to an intermediate volume.

    Args:
        acc_chunks (list): Accumulated chunks
        embeddings (list): Accumulated embeddings
        batch_size (int): Batch size
    """
    import pyarrow as pa
    from datasets import Dataset

    table = pa.Table.from_arrays(
        [
            pa.array([chunk[0] for chunk in acc_chunks]),  # id
            pa.array([chunk[1] for chunk in acc_chunks]),  # url
            pa.array([chunk[2] for chunk in acc_chunks]),  # title
            pa.array([chunk[3] for chunk in acc_chunks]),  # text
            pa.array(acc_embeddings),
        ],
        names=["id", "url", "title", "text", "embedding"],
    )
    path_parent_folder = f"{INTERMEDIATE_DIR}/{MODEL_SLUG}-{batch_size}"
    dataset = Dataset(table)
    dataset.save_to_disk(path_parent_folder)
    EMBEDDING_INTERMEDIATE_VOLUME.commit()
    print(f"Saved at {path_parent_folder}")