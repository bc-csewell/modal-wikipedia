# Modal Wikipedia Embedding kata

Following the blog post [here](https://modal.com/blog/embedding-wikipedia?utm_source=tldrai) using Modal to create embeddings the an opensource wikipedia dataset. Requires a modal account + cli to be installed & a hugging face account.

To download the dataset
```
modal run download.py::download_dataset  
```
To create embeddings
```
modal run main.py::embed_dataset 
```
TO DO, To upload to hugging face