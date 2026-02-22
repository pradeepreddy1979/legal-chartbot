# ⚖️ LegalBot — AI-Powered Legal Reference System

An interactive Streamlit application that helps users find relevant Indian legal sections and short explanations using semantic search + LLM assistance.

**Status:** Active development. This README summarizes how to set up, run, and extend the project.

---

## **Project Overview**
- **Purpose:** Assist students, lawyers, and individuals to quickly locate and understand Indian legal sections (IPC, CrPC, and other acts).
- **Architecture:** Local vector store  for semantic search (embeddings), optional LLM generation (Gemini), and a Streamlit chat UI.

---

## **Core Features**
- **Semantic Search:** Local embeddings using `sentence-transformers` to retrieve relevant legal sections.
- **LLM Responses:** Optional generation via Gemini (configured with `GEMINI_API_KEY` environment variable). The app has robust retry/fallback behavior.
- **Streamlit UI:** Chat-style interface with quick actions and feedback buttons (feedback stored in SQLite `user_feedback.db`).
- **Document Vault:** Upload PDFs for analysis and local indexing utilities in `create_embeddings.py` and `pdf_extraction.py`.

---

## **Quick Start (Windows / PowerShell)**
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with required keys (example):

```text
GEMINI_API_KEY=your_gemini_api_key_here
# Optional Google OAuth (if configured):
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
```

4. Initialize the database (optional - the app will create tables automatically on first run):

```powershell
python db_manager.py
```

5. Run the app:

```powershell
streamlit run app.py
```

Open `http://localhost:8502` (or the port printed by Streamlit) in your browser.

---

## **Auth & Users**
- Local email/password authentication is implemented. Credentials are stored hashed in `user_feedback.db`.
- A simplified Google Sign-In flow is available in the UI (server-side OAuth is not fully implemented); see `GOOGLE_OAUTH_SETUP.md` for guidance if you want production OAuth.

---

## **Feedback & Logs**
- User feedback is logged to the SQLite database (`user_feedback.db`) via `db_manager.log_interaction`.
- The UI provides quick feedback buttons; feedback is stored server-side and is not exposed to end users.

---

## **Important Files**
- `app.py` — Main Streamlit app.
- `db_manager.py` — Database helpers and user/feedback management.
- `create_embeddings.py` — Script to index documents and create embeddings.
- `pdf_extraction.py` — Utilities for extracting text from PDFs.
- `legal_vector_db.json` — Local vector database used by the app (embeddings + metadata).
- `user_feedback.db` — SQLite DB storing users and feedback.
- `data/` — Raw documents and indexed JSON.
- `extracted_data/` — Extracted legal text JSON files.

---

## **Developer Notes**
- The app uses `sentence-transformers` (`all-MiniLM-L6-v2`) for local embeddings and `sklearn` cosine similarity for retrieval.
- Gemini integration is optional — the app falls back to showing retrieved sections if the LLM is unavailable.
- To re-index documents, run `python create_embeddings.py` after placing source PDFs in `data/acts/`.

---

## **Testing & Maintenance**
- Quick tests: `test_db.py`, `test_gemini_api.py`, and `test_render.py` are present as lightweight checks — run them individually.
- Remove caches with `git clean -fdX` (careful: this deletes ignored files).

---

## **Help & Contact**
If you need help setting up Gemini or OAuth, open an issue or contact the repo maintainer.

---

Thank you for using LegalBot — contributions welcome!


## 🚀 Future Scope

- Support for multiple Indian languages.  
- Integration of case law retrieval and judgments.  
- Voice-based legal assistance.  
- Smart search filters for faster query responses.

---

### ⚖️ “Simplifying access to Indian law through the power of AI.”
