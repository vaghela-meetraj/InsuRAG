# PolicyScan

An Agentic Retrieval-Augmented Generation (RAG) assistant for querying and extracting information from health insurance policy documents.

---

## What Has Been Done So Far

1. **Document Loading**: Ingested and processed raw health insurance policy contracts from `data/policy.txt` using LangChain's `TextLoader`.
2. **Chunking Strategy**: Split the text using `RecursiveCharacterTextSplitter` with a chunk size of 1,000 characters and an overlap of 150 characters to preserve context across boundaries.
3. **Embedding Generation**: Embedded text chunks using Jina AI's `jina-embeddings-v2-base-en` model.
4. **Vector Database**: Stored and indexed chunk embeddings locally using Facebook AI Similarity Search (`FAISS`).
5. **Retriever Tuning**: Configured similarity search with $k=8$ to ensure high recall for specific policy sub-limits (e.g., room rent, ambulance, cataract surgery).
6. **Tool Integration**: Created a dedicated tool `search_insurance_policy` allowing the model to query the FAISS index on demand.
7. **Agent Setup**: Configured an autonomous agent using `create_agent` powered by Groq (`openai/gpt-oss-120b`).
8. **Anti-Hallucination Guardrails**: Implemented system instructions ensuring the agent only answers using retrieved context and refuses to fabricate missing information.
9. **CLI Interface**: Packaged the end-to-end pipeline into `main.py` with an `ask_insurance_assistant` function for querying.

---

## Steps to Set Up and Run

### 1. Clone the Repository
```bash
git clone https://github.com/vaghela-meetraj/PolicyScan
cd PolicyScan
```

### 2. Set Up Virtual Environment
```powershell
# Using uv (fast)
uv venv virtenv
.\virtenv\Scripts\activate

# Or using standard python
python -m venv virtenv
.\virtenv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory (refer to `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
```

### 5. Run the Application
```powershell
python main.py
```
