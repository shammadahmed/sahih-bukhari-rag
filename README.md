# Sahih al-Bukhari + Your Documents — Streamlit RAG Assistant

This Streamlit app ships with a **prebuilt SQLite FTS5/BM25 index** generated offline from the nine unique EPUB volumes provided for this project. The preloaded corpus is ready immediately: the app does not extract EPUBs, chunk the books, generate document embeddings, download an ML model, train a model, or rebuild the bundled index at startup.

It also supports **on-the-fly uploads**. A user can upload TXT, Markdown, CSV, text-based PDF, or DOCX files in the sidebar, click **Add files to this chat**, and search those documents alongside the preloaded books. Uploaded documents are chunked into overlapping passages and indexed in an in-memory SQLite FTS5 database scoped to that Streamlit browser session. They do not modify `bukhari_fts.db` and are cleared when the session ends or the user clicks **Clear uploaded documents**.

## Included files

- `app.py` — Streamlit UI, preloaded-book search, temporary uploaded-document indexing, retrieval fusion, and Groq answer generation.
- `bukhari_fts.db` — prebuilt index with 4,488 multi-hadith passages from nine unique volumes.
- `index_manifest.json` — metadata about the prebuilt index.
- `requirements.txt` — Streamlit, Groq, PDF text extraction, and DOCX parsing dependencies.

## Supported uploads

- `.txt`, `.md`, `.csv` — decoded as text.
- `.pdf` — extracts selectable text from PDF pages and records page ranges. Scanned/image-only PDFs are not OCR-processed; they should be OCR'd before upload or supplied as text.
- `.docx` — extracts paragraphs and tables. DOCX page numbers are not available, so citations identify the filename and passage number.

Each file is limited to 20 MiB, with up to 10 files per upload batch. Uploaded documents are split into roughly 350-word passages with 60-word overlap. This happens only after the user clicks **Add files to this chat**, not when the site opens. Retrieval uses SQLite FTS5/BM25 and reciprocal-rank fusion to combine results from the preloaded books and uploaded documents.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, `bukhari_fts.db`, and `index_manifest.json` to a GitHub repository.
2. Create a Streamlit Community Cloud app with `app.py` as the entrypoint.
3. In app **Settings → Secrets**, add:

   ```toml
   Groq_key = "your_groq_api_key_here"
   ```

4. Deploy. The preloaded books are ready when the app opens. Visitors can add their own temporary documents from the sidebar.

Do not commit a real API key. If other people can access the app while it uses your Groq key, their questions can consume your API quota; add access controls or keep the app private if needed.

## Storage and privacy behavior

The bundled Sahih al-Bukhari index is read-only. Uploaded text is held in an in-memory SQLite FTS5 index associated with the active Streamlit session. The app does not write uploaded documents to the project directory or append them to the bundled index. Session data is temporary and may be lost when the session ends or the hosting process restarts. Uploaded text passages selected for an answer are sent to Groq's API as context for answer generation.

## Retrieval limitations

This is lexical retrieval (SQLite FTS5/BM25), not dense-vector semantic search. It is lightweight and needs no embedding model, but it may miss relevant passages when a query uses terms that differ substantially from the wording in the source. OCR defects in the source books may remain. Verify important answers against the original edition.

The passage page labels for the preloaded books refer to internal EPUB page identifiers, not necessarily printed page numbers.

## Copyright

The bundled `.db` contains extracted book passages. Keep the repository private, or obtain permission before redistributing the underlying translation or its extracted content.
