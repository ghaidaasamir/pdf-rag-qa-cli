# PDF Q&A CLI App

A command-line tool that lets you ask questions about any PDF using RAG (Retrieval Augmented Generation).


## Run it

You need [Ollama](https://ollama.com) running with a model pulled (the script
uses `gemma3:4b`).

```bash
pip install -r requirements.txt
ollama pull gemma3:4b
python pdf_qa.py path/to/your.pdf
```

Then just type questions. `quit` / `exit` / `q` to stop.

## Notes

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Answers stream token by token.
- First run downloads the embedding model, so give it a minute.
