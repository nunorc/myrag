
# MyRAG

Bootstraping a minimalist RAG based question-answering application to run locally.

## Quickstart

Build the required images using Docker compose:

```bash
$ docker compose build
```

Run the services:

```bash
$ docker compose up -d
```

Copy the documents to a folder, e.g. `docs` and run the `import-docs.py`:

```bash
$ python import-docs.py docs
```

Visit the app from a browser `http://localhost:8501` to make queries.
Or perform query from python:

```python
from myrag import MyRAG

rag = MyRAG()
response_text, data = rag.query("question text")
  # returns the response text and the data including the text, the sources and the context
```

## Acknowledgements

* [Chroma](https://www.trychroma.com/)
* [Ollama](https://ollama.com/)
* [LangChain](https://www.langchain.com/)
* [Let's build a RAG system - The Ollama Course](https://www.youtube.com/watch?v=FQTCLOUnIzI) by [Matt Williams](https://www.youtube.com/@technovangelist)
* [Python RAG Tutorial (with Local LLMs): AI For Your PDFs](https://www.youtube.com/watch?v=2TJxpyO3ei4) by [pixegami](https://www.youtube.com/@pixegami)
