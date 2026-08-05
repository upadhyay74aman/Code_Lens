# HOW TO RUN CODELENS
# 1. pip install -r requirements.txt
# 2. Create .env file with: GEMINI_API_KEY=your_key
#    Get free key at: https://aistudio.google.com/apikey
# 3. Run: uvicorn main:app --reload --port 8000
# 4. Open http://localhost:8000 in your browser
# 5. Paste any public GitHub repo URL and click Index Repository

import os
import shutil
import stat
import tempfile
import faiss
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI(title="CodeLens API", description="AI Codebase Intelligence Engine Backend")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory state
indexed_repo_name: Optional[str] = None
indexed_repo_url: Optional[str] = None
indexed_branch: Optional[str] = None
all_file_paths: List[str] = []
file_tree: List[Dict[str, Any]] = []
faiss_index: Optional[faiss.IndexFlatIP] = None
chunk_metadata: List[Dict[str, Any]] = [] # parallel to FAISS index
chunk_contents: List[str] = [] # parallel to FAISS index
repo_summary: str = ""
is_indexed: bool = False


class IndexRequest(BaseModel):
    repo_url: str
    branch: str = "main"


class QueryRequest(BaseModel):
    question: str


class ChatMessage(BaseModel):
    role: str  # "user" or "model" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]

def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to remove read-only files on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def safe_rmtree(path):
    """Safely removes a directory tree, resolving read-only file issues on Windows."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=remove_readonly)
        except Exception:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass


def get_repo_name(url: str) -> str:
    """Extracts the repository name from a GitHub URL."""
    url_stripped = url.rstrip("/")
    if url_stripped.endswith(".git"):
        name = url_stripped.split("/")[-1][:-4]
    else:
        name = url_stripped.split("/")[-1]
    return name


def build_file_tree(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Builds a nested JSON file tree from a flat list of file paths."""
    root: List[Dict[str, Any]] = []
    for path in sorted(file_paths):
        parts = path.split('/')
        current_level = root
        current_path = ""
        for i, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path else part
            is_last = (i == len(parts) - 1)
            node_type = "file" if is_last else "directory"
            
            # Search for an existing node at this level
            existing_node = next((node for node in current_level if node["name"] == part), None)
            
            if not existing_node:
                new_node = {
                    "name": part,
                    "type": node_type,
                    "path": current_path
                }
                if not is_last:
                    new_node["children"] = []
                current_level.append(new_node)
                current_level = new_node.get("children", [])
            else:
                current_level = existing_node.get("children", [])
    return root


def generate_content_with_fallback(contents, system_instruction=None):
    """Generates content using Gemini models, trying multiple models in case of quota exhaustion."""
    models_to_try = ["models/gemini-2.0-flash", "models/gemini-2.5-flash"]
    last_err = None
    for model_name in models_to_try:
        try:
            print(f"[CodeLens] Trying content generation with {model_name}...")
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
            else:
                model = genai.GenerativeModel(model_name=model_name)
                
            response = model.generate_content(contents)
            return response.text.strip()
        except Exception as e:
            print(f"[CodeLens] Model {model_name} failed: {e}")
            last_err = e
            continue
    raise last_err


def get_parser_binary() -> str:
    """Finds or compiles the Go parser binary, returns its path."""
    import subprocess
    import sys
    import shutil
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    binary_name = "codelens-parser.exe" if sys.platform.startswith("win") else "codelens-parser"
    binary_path = os.path.join(dir_path, binary_name)
    
    if os.path.exists(binary_path):
        return binary_path
        
    # If not found, check if 'go' is in PATH to compile it
    go_path = shutil.which("go")
    if go_path:
        print(f"[CodeLens] Binary {binary_name} not found, compiling using system Go...")
        try:
            main_go_path = os.path.join(dir_path, "parser", "main.go")
            subprocess.run([go_path, "build", "-o", binary_path, main_go_path], check=True)
            return binary_path
        except Exception as e:
            print(f"[CodeLens] Dynamic compilation failed: {e}")
            
    # Also check if we have local go_dist
    local_go = os.path.join(dir_path, "go_dist", "go", "bin", "go" + (".exe" if sys.platform.startswith("win") else ""))
    if os.path.exists(local_go):
        print(f"[CodeLens] Binary not found, compiling using local Go distribution...")
        try:
            main_go_path = os.path.join(dir_path, "parser", "main.go")
            subprocess.run([local_go, "build", "-o", binary_path, main_go_path], check=True)
            return binary_path
        except Exception as e:
            print(f"[CodeLens] Local compilation failed: {e}")
            
    raise RuntimeError(
        f"Go parser binary not found at {binary_path} and could not compile it. "
        "Please compile parser/main.go or make sure Go is installed."
    )


@app.post("/api/index")
async def index_repository(request: IndexRequest):
    """Clones a GitHub repository, processes text files, generates embeddings, and creates a search index."""
    global indexed_repo_name, indexed_repo_url, indexed_branch, all_file_paths, file_tree, faiss_index, chunk_metadata, chunk_contents, repo_summary, is_indexed
    
    import subprocess
    import json
    
    # Reload/configure API key dynamically in case it was added to .env after server startup
    api_key_check = os.getenv("GEMINI_API_KEY")
    if not api_key_check:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not configured in .env file.")
    genai.configure(api_key=api_key_check)

    repo_name = get_repo_name(request.repo_url)
    print(f"[CodeLens] Starting indexing process for repo: {repo_name} (branch: {request.branch})")

    # Setup temporary directory for clone
    temp_dir = os.path.join(tempfile.gettempdir(), 'codelens', repo_name)
    print(f"[CodeLens] Cleaning existing temp folder: {temp_dir}")
    safe_rmtree(temp_dir)

    # 1. Locate and run Go parser CLI
    try:
        binary_path = get_parser_binary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    print(f"[CodeLens] Invoking Go parser for {request.repo_url}...")
    cmd = [
        binary_path,
        "--repo-url", request.repo_url,
        "--branch", request.branch,
        "--temp-dir", temp_dir
    ]

    try:
        # Run Go parser CLI
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    except Exception as e:
        safe_rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to execute Go parser binary: {e}")

    if result.returncode != 0 or not result.stdout:
        safe_rmtree(temp_dir)
        err_msg = result.stderr or f"Go parser exited with code {result.returncode}"
        raise HTTPException(status_code=500, detail=f"Go parser execution error: {err_msg}")

    try:
        data = json.loads(result.stdout)
    except Exception as e:
        safe_rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Go parser returned invalid JSON: {result.stdout[:500]}")

    if data.get("status") == "error":
        safe_rmtree(temp_dir)
        raise HTTPException(status_code=400, detail=data.get("detail", "Unknown error during parsing."))

    # 2. Extract Go parser output
    local_chunks = [c["content"] for c in data["chunks"]]
    local_meta = [{
        "file_path": c["file_path"],
        "start_line": c["start_line"],
        "language": c["language"],
        "chunk_index": c["chunk_index"]
    } for c in data["chunks"]]

    files_indexed = data["files_indexed"]
    chunks_created = data["chunks_created"]
    file_tree_data = data["file_tree"]
    readme_snippet = data["readme_snippet"]

    relative_paths = sorted(list(set(c["file_path"] for c in data["chunks"])))

    # 3. Generate embeddings using Gemini
    print("[CodeLens] Generating embeddings using Gemini models/embedding-001...")
    embeddings = []
    batch_size = 100
    for i in range(0, len(local_chunks), batch_size):
        batch = local_chunks[i:i+batch_size]
        try:
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(res['embedding'])
            print(f"[CodeLens] Embedded chunks {i} to {min(i+batch_size, len(local_chunks))}...")
        except Exception as embed_err:
            print(f"[CodeLens] Gemini embedding error: {embed_err}")
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API error during embedding generation: {str(embed_err)}"
            )

    # 4. Build FAISS index (inner product flat index)
    print("[CodeLens] Building FAISS search index...")
    dimension = len(embeddings[0])
    vectors = np.array(embeddings).astype('float32')
    faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity (Inner Product)
    
    new_faiss_index = faiss.IndexFlatIP(dimension)
    new_faiss_index.add(vectors)

    # 5. Generate repository summary
    print("[CodeLens] Requesting codebase high-level summary from Gemini...")
    file_list_str = "\n".join(relative_paths[:150]) # Limiting file list in prompt to prevent token limit
    if len(relative_paths) > 150:
        file_list_str += f"\n... and {len(relative_paths) - 150} other files."

    summary_prompt = f"Here is the list of files in the repository:\n{file_list_str}\n\n"
    if readme_snippet:
        summary_prompt += f"Here is the first 3000 characters of the repository README.md:\n{readme_snippet}\n\n"
    summary_prompt += "Summarize this codebase in exactly 3 sentences. What does it do, what stack does it use, and what are the main components?"

    try:
        summary_text = generate_content_with_fallback(summary_prompt)
    except Exception as sum_err:
        print(f"[CodeLens] Warning: Gemini codebase summary generation failed - {sum_err}")
        summary_text = f"This codebase contains {len(relative_paths)} files. Stack components include languages like " + \
                       f"{', '.join(list(set(local_meta[i]['language'] for i in range(len(local_meta))))[:4])}."

    # Clean up temp folder
    safe_rmtree(temp_dir)

    # Commit to global state
    indexed_repo_name = repo_name
    indexed_repo_url = request.repo_url
    indexed_branch = request.branch
    all_file_paths = relative_paths
    file_tree = file_tree_data
    faiss_index = new_faiss_index
    chunk_metadata = local_meta
    chunk_contents = local_chunks
    repo_summary = summary_text
    is_indexed = True

    print(f"[CodeLens] Indexing complete! Indexed {files_indexed} files, {chunks_created} chunks.")

    return {
        "status": "success",
        "files_indexed": files_indexed,
        "chunks_created": chunks_created,
        "repo_summary": repo_summary,
        "file_tree": file_tree
    }


@app.post("/api/query")
async def query_codebase(request: QueryRequest):
    """Performs semantic search against the FAISS index and returns a detailed answer using Gemini 2.5 Flash."""
    global faiss_index, chunk_contents, chunk_metadata, is_indexed
    
    if not is_indexed or faiss_index is None:
        raise HTTPException(status_code=400, detail="No repository has been indexed yet. Please index a codebase first.")

    print(f"[CodeLens] Processing query: {request.question}")
    
    # 1. Embed query
    try:
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=request.question,
            task_type="retrieval_query"
        )
        query_vector = res['embedding']
    except Exception as embed_err:
        print(f"[CodeLens] Gemini embedding error: {embed_err}")
        raise HTTPException(status_code=500, detail=f"Gemini API error during embedding: {str(embed_err)}")

    # 2. Search FAISS index
    q_vec = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(q_vec)
    
    k = min(8, len(chunk_contents))
    D, I = faiss_index.search(q_vec, k)

    # Extract retrieved chunks
    retrieved_chunks = []
    sources = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        content = chunk_contents[idx]
        meta = chunk_metadata[idx]
        score = float(D[0][rank])
        
        retrieved_chunks.append({
            "content": content,
            "file_path": meta["file_path"],
            "start_line": meta["start_line"]
        })
        
        sources.append({
            "file": meta["file_path"],
            "preview": content[:120],
            "start_line": meta["start_line"],
            "score": score
        })

    # 3. Construct prompt
    chunks_formatted = ""
    for chunk in retrieved_chunks:
        chunks_formatted += f"--- [{chunk['file_path']}] ---\n{chunk['content']}\n\n"

    system_prompt = (
        "You are CodeLens, an expert AI assistant that deeply understands codebases.\n"
        "You answer questions about code with precision, always citing exact file paths "
        "and line numbers. Format your answers clearly with:\n"
        "- A direct answer first\n"
        "- Relevant code snippets in markdown code blocks with language tags\n"
        "- File paths cited as [filename.py]\n"
        "- If you find multiple relevant locations, list all of them\n"
        "- End with a 'Related files you might want to check:' section"
    )

    user_prompt = (
        f"Here are the most relevant code chunks from the repository:\n\n"
        f"{chunks_formatted}"
        f"Question: {request.question}\n\n"
        f"Answer thoroughly, cite specific files and explain the code clearly."
    )

    try:
        answer_text = generate_content_with_fallback(user_prompt, system_instruction=system_prompt)
    except Exception as gen_err:
        print(f"[CodeLens] Gemini generation error: {gen_err}")
        raise HTTPException(status_code=500, detail=f"Gemini API error during generation: {str(gen_err)}")

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(retrieved_chunks)
    }


@app.get("/api/filetree")
async def get_file_tree():
    """Returns the nested file tree of the currently indexed repository."""
    if not is_indexed:
        raise HTTPException(status_code=400, detail="No repository has been indexed yet.")
    return file_tree


@app.get("/api/status")
async def get_indexing_status():
    """Returns statistics about the current indexed repository."""
    return {
        "indexed": is_indexed,
        "repo_name": indexed_repo_name or "",
        "chunks": len(chunk_contents) if is_indexed else 0,
        "files": len(all_file_paths) if is_indexed else 0
    }


@app.post("/api/chat")
async def chat_conversation(request: ChatRequest):
    """Conversation mode carrying context over multiple rounds using sliding window context and RAG search."""
    global faiss_index, chunk_contents, chunk_metadata, is_indexed
    
    if not is_indexed or faiss_index is None:
        raise HTTPException(status_code=400, detail="No repository has been indexed yet. Please index a codebase first.")

    # 1. Slide window to last 10 messages
    history = request.messages[-10:]
    
    # 2. Find the last user message to extract question and perform RAG retrieval
    last_user_msg = None
    for msg in reversed(history):
        if msg.role.lower() in ("user", "human"):
            last_user_msg = msg
            break
            
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in chat history.")

    print(f"[CodeLens Chat] Processing conversation query: {last_user_msg.content}")

    # 3. Retrieve relevant chunks for latest question
    try:
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=last_user_msg.content,
            task_type="retrieval_query"
        )
        query_vector = res['embedding']
    except Exception as embed_err:
        print(f"[CodeLens] Gemini embedding error: {embed_err}")
        raise HTTPException(status_code=500, detail=f"Gemini API error during embedding: {str(embed_err)}")

    q_vec = np.array([query_vector]).astype('float32')
    faiss.normalize_L2(q_vec)
    
    k = min(8, len(chunk_contents))
    D, I = faiss_index.search(q_vec, k)

    retrieved_chunks = []
    sources = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        content = chunk_contents[idx]
        meta = chunk_metadata[idx]
        score = float(D[0][rank])
        
        retrieved_chunks.append({
            "content": content,
            "file_path": meta["file_path"],
            "start_line": meta["start_line"]
        })
        
        sources.append({
            "file": meta["file_path"],
            "preview": content[:120],
            "start_line": meta["start_line"],
            "score": score
        })

    chunks_formatted = ""
    for chunk in retrieved_chunks:
        chunks_formatted += f"--- [{chunk['file_path']}] ---\n{chunk['content']}\n\n"

    # 4. Construct rewritten user message prompt
    rewritten_prompt = (
        f"Here are the most relevant code chunks from the repository:\n\n"
        f"{chunks_formatted}"
        f"Question: {last_user_msg.content}\n\n"
        f"Answer thoroughly, cite specific files and explain the code clearly."
    )

    # 5. Format history for Gemini
    gemini_contents = []
    for msg in history:
        # Map assistant role to model
        role = "model" if msg.role.lower() in ("assistant", "model") else "user"
        
        text = msg.content
        if msg is last_user_msg:
            # Inject RAG context into the last query
            text = rewritten_prompt
            
        gemini_contents.append({
            "role": role,
            "parts": [{"text": text}]
        })

    system_prompt = (
        "You are CodeLens, an expert AI assistant that deeply understands codebases.\n"
        "You answer questions about code with precision, always citing exact file paths "
        "and line numbers. Format your answers clearly with:\n"
        "- A direct answer first\n"
        "- Relevant code snippets in markdown code blocks with language tags\n"
        "- File paths cited as [filename.py]\n"
        "- If you find multiple relevant locations, list all of them\n"
        "- End with a 'Related files you might want to check:' section"
    )

    try:
        answer_text = generate_content_with_fallback(gemini_contents, system_instruction=system_prompt)
    except Exception as gen_err:
        print(f"[CodeLens] Gemini generation error: {gen_err}")
        raise HTTPException(status_code=500, detail=f"Gemini API error during generation: {str(gen_err)}")

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(retrieved_chunks)
    }


@app.get("/")
async def serve_index():
    """Serves the index.html frontend."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
