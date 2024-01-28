from modal import Volume

DATASET_READ_VOLUME = Volume.persisted("wikipedia")
EMBEDDING_INTERMEDIATE_VOLUME = Volume.persisted("embeddings")
DATASET_DIR = "/data"
INTERMEDIATE_DIR = "/embeddings"

MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 768
MODEL_SLUG = MODEL_ID.split("/")[-1]
