# HOW TO RUN CODELENS
# 1. pip install -r requirements.txt
# 2. Create .env file with: GEMINI_API_KEY=your_key
#    Get free key at: https://aistudio.google.com/apikey
# 3. Run: uvicorn main:app --reload --port 8000
# 4. Open index.html in browser (use Live Server in VS Code)
# 5. Paste any public GitHub repo URL and click Index Repository

import os
import shutil
import tempfile
import git
import faiss
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


import stat

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
    models_to_try = ["models/gemini-flash-latest", "models/gemini-2.0-flash", "models/gemini-2.5-flash"]
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


@app.post("/api/index")
async def index_repository(request: IndexRequest):
    """Clones a GitHub repository, processes text files, generates embeddings, and creates a search index."""
    global indexed_repo_name, indexed_repo_url, indexed_branch, all_file_paths, file_tree, faiss_index, chunk_metadata, chunk_contents, repo_summary, is_indexed
    
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

    # 1. Clone repository
    print(f"[CodeLens] Cloning repository: {request.repo_url} (branch: {request.branch})")
    try:
        # Clone repo using GitPython
        git.Repo.clone_from(request.repo_url, temp_dir, branch=request.branch)
    except git.exc.GitCommandError as e:
        print(f"[CodeLens] Git clone error: {e}")
        # Clean up directory on failure
        safe_rmtree(temp_dir)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository. Make sure the repository is public and the branch exists. Error details: {str(e)}"
        )
    except Exception as e:
        print(f"[CodeLens] Unexpected clone error: {e}")
        safe_rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Unexpected error during git clone: {str(e)}")

    # 2. Walk files recursively matching allowed extensions
    skip_dirs = {"node_modules", ".git", "__pycache__", "dist", "build", ".next", "venv", "env"}
    allowed_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', 
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.md'
    }
    
    print("[CodeLens] Reading repository files...")
    file_list = []
    for root, dirs, files in os.walk(temp_dir):
        # Modify dirs in-place to prevent os.walk from entering skipped directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_extensions:
                file_list.append(os.path.join(root, file))

    if not file_list:
        safe_rmtree(temp_dir)
        raise HTTPException(
            status_code=400, 
            detail="No files matching the supported extensions found in this repository (.py, .js, .ts, etc.)."
        )

    print(f"[CodeLens] Found {len(file_list)} matching files to index.")

    # 3. Read and split files
    print("[CodeLens] Splitting files into character chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    local_chunks = []
    local_meta = []
    
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.md': 'markdown'
    }

    for full_path in file_list:
        rel_path = os.path.relpath(full_path, temp_dir).replace('\\', '/')
        ext = os.path.splitext(full_path)[1].lower()
        lang = extension_map.get(ext, 'text')
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as file_err:
            print(f"[CodeLens] Warning: Failed to read {rel_path} - {file_err}")
            continue
            
        file_chunks = splitter.split_text(content)
        
        last_pos = 0
        for idx, chunk_text in enumerate(file_chunks):
            # Calculate the line number of this chunk in the original file
            pos = content.find(chunk_text, last_pos)
            if pos == -1:
                pos = content.find(chunk_text)
                
            if pos != -1:
                start_line = content[:pos].count('\n') + 1
                last_pos = pos + len(chunk_text)
            else:
                start_line = 1
                
            local_chunks.append(chunk_text)
            local_meta.append({
                "file_path": rel_path,
                "start_line": start_line,
                "language": lang,
                "chunk_index": idx
            })

    if not local_chunks:
        raise HTTPException(status_code=400, detail="Failed to parse text from any repository files.")

    print(f"[CodeLens] Created {len(local_chunks)} text chunks.")

    # 4. Generate embeddings using Gemini
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

    # 5. Build FAISS index (inner product flat index)
    print("[CodeLens] Building FAISS search index...")
    dimension = len(embeddings[0])
    vectors = np.array(embeddings).astype('float32')
    faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity (Inner Product)
    
    new_faiss_index = faiss.IndexFlatIP(dimension)
    new_faiss_index.add(vectors)

    # 6. Generate repository summary
    print("[CodeLens] Requesting codebase high-level summary from Gemini...")
    readme_path = None
    for file in os.listdir(temp_dir):
        if file.lower() == "readme.md":
            readme_path = os.path.join(temp_dir, file)
            break
            
    readme_snippet = ""
    if readme_path and os.path.isfile(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                readme_snippet = f.read(3000)
        except Exception as readme_err:
            print(f"[CodeLens] Warning: Could not read README.md - {readme_err}")

    # Build relative paths list
    relative_paths = [os.path.relpath(f, temp_dir).replace('\\', '/') for f in file_list]
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

    # Build file tree structure
    tree = build_file_tree(relative_paths)

    # Commit to global state
    indexed_repo_name = repo_name
    indexed_repo_url = request.repo_url
    indexed_branch = request.branch
    all_file_paths = relative_paths
    file_tree = tree
    faiss_index = new_faiss_index
    chunk_metadata = local_meta
    chunk_contents = local_chunks
    repo_summary = summary_text
    is_indexed = True

    print(f"[CodeLens] Indexing complete! Indexed {len(file_list)} files, {len(local_chunks)} chunks.")

    return {
        "status": "success",
        "files_indexed": len(file_list),
        "chunks_created": len(local_chunks),
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
