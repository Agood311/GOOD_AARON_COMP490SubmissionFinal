# main.py
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("rfp.csv")
TEMPLATES_DIR = Path("templates")

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Globals
df: pd.DataFrame | None = None
texts: List[str] | None = None
tfidf_vectorizer: TfidfVectorizer | None = None
tfidf_matrix = None
embed_model: SentenceTransformer | None = None
embeddings: np.ndarray | None = None


def init_indexes() -> None:
    """
    Load rfp.csv, clean, and build:
      - combined_text
      - TF-IDF matrix
      - semantic embeddings
    """
    global df, texts, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings

    if not DATA_FILE.exists():
        raise RuntimeError("rfp.csv not found. Run ingest_sam.py first.")

    df_loaded = pd.read_csv(DATA_FILE, dtype=str).fillna("")

    if "id" not in df_loaded.columns:
        raise RuntimeError("rfp.csv must contain an 'id' column")

    # Deduplicate by id
    df_loaded = df_loaded.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Ensure columns exist
    for col in [
        "title",
        "description_text",
        "organization_name",
        "full_parent_path_name",
        "response_date",
        "ui_link",
        "source_url",
        "additional_info_link",
        "naics",
        "psc",
        "state",
        "place_of_performance",
    ]:
        if col not in df_loaded.columns:
            df_loaded[col] = ""

    # Build combined text
    combined_texts: list[str] = []
    for _, row in df_loaded.iterrows():
        pieces = []

        title = row.get("title", "").strip()
        if title:
            pieces.append(title)

        org = row.get("organization_name", "").strip()
        if org:
            pieces.append(org)

        parent = row.get("full_parent_path_name", "").strip()
        if parent:
            pieces.append(parent)

        naics = row.get("naics", "").strip()
        if naics:
            pieces.append(f"NAICS {naics}")

        psc = row.get("psc", "").strip()
        if psc:
            pieces.append(f"PSC {psc}")

        desc = row.get("description_text", "").strip()
        if desc:
            pieces.append(desc)

        combined_texts.append(" ".join(pieces))

    df_loaded["combined_text"] = combined_texts

    # Drop rows with no text
    mask = df_loaded["combined_text"].str.len() > 0
    df_loaded = df_loaded[mask].reset_index(drop=True)

    if df_loaded.empty:
        raise RuntimeError("No usable rows with text found in rfp.csv")

    combined = df_loaded["combined_text"].tolist()

    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(combined)

    # Build semantic embeddings
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(
        combined,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    df = df_loaded
    texts = combined
    tfidf_vectorizer = vectorizer
    tfidf_matrix = X
    embed_model = model
    embeddings = emb

    globals()["df"] = df
    globals()["texts"] = texts
    globals()["tfidf_vectorizer"] = tfidf_vectorizer
    globals()["tfidf_matrix"] = tfidf_matrix
    globals()["embed_model"] = embed_model
    globals()["embeddings"] = embeddings

    print(f"Initialized indexes on {len(df)} RFPs")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def run_tfidf(query: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, normalized_scores) for TF-IDF search."""
    q_vec = tfidf_vectorizer.transform([query])
    sims = linear_kernel(q_vec, tfidf_matrix).ravel()
    # Get top-k indices
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_idx]
    # Filter out zero scores
    mask = top_scores > 0
    return top_idx[mask], normalize_scores(top_scores[mask])


def run_semantic(query: str, top_k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, normalized_scores) for semantic search."""
    q_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_vec  # cosine similarity (already normalized)
    # Get top-k indices
    top_idx = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_idx]
    # Filter out zero/negative scores
    mask = top_scores > 0
    return top_idx[mask], normalize_scores(top_scores[mask])


def run_hybrid(query: str, alpha: float = 0.5, top_k: int = 100) -> pd.DataFrame:
    """
    Hybrid search combining TF-IDF and semantic scores.
    
    Final score = alpha * tfidf_normalized + (1 - alpha) * semantic_normalized
    
    Args:
        query: Search query string
        alpha: Weight for TF-IDF (0.0 = pure semantic, 1.0 = pure TF-IDF)
        top_k: Number of results to return
    
    Returns:
        DataFrame with results sorted by hybrid score
    """
    # Get results from both methods
    tfidf_idx, tfidf_scores = run_tfidf(query, top_k=top_k * 2)
    sem_idx, sem_scores = run_semantic(query, top_k=top_k * 2)
    
    # Build score dictionaries
    tfidf_dict = dict(zip(tfidf_idx, tfidf_scores))
    sem_dict = dict(zip(sem_idx, sem_scores))
    
    # Get union of all candidate indices
    all_idx = set(tfidf_idx) | set(sem_idx)
    
    # Compute hybrid scores
    hybrid_scores = {}
    for idx in all_idx:
        t_score = tfidf_dict.get(idx, 0.0)
        s_score = sem_dict.get(idx, 0.0)
        hybrid_scores[idx] = alpha * t_score + (1 - alpha) * s_score
    
    # Sort by hybrid score
    sorted_idx = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:top_k]
    
    # Build result DataFrame
    out = df.iloc[sorted_idx].copy()
    out["score"] = [hybrid_scores[i] for i in sorted_idx]
    out["tfidf_score"] = [tfidf_dict.get(i, 0.0) for i in sorted_idx]
    out["semantic_score"] = [sem_dict.get(i, 0.0) for i in sorted_idx]
    
    return out


def run_search(query: str, mode: str, top_k: int = 50) -> pd.DataFrame:
    """Unified search interface."""
    if mode == "tfidf":
        idx, scores = run_tfidf(query, top_k)
        out = df.iloc[idx].copy()
        out["score"] = scores
        return out
    elif mode == "semantic":
        idx, scores = run_semantic(query, top_k)
        out = df.iloc[idx].copy()
        out["score"] = scores
        return out
    else:  # hybrid
        return run_hybrid(query, alpha=0.5, top_k=top_k)


def parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse various date formats."""
    if not date_str:
        return None
    # Common formats from SAM.gov
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%Y%m%d"]:
        try:
            return datetime.strptime(date_str.split("T")[0].split(" ")[0], fmt)
        except ValueError:
            continue
    return None


def apply_filters(
    df_res: pd.DataFrame,
    state: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Apply optional filters to search results."""
    if df_res.empty:
        return df_res
    
    # State filter
    if state:
        state = state.upper().strip()
        df_res = df_res[df_res["state"].str.upper().str.strip() == state]
    
    # Date range filter
    if date_from or date_to:
        date_from_dt = parse_date(date_from) if date_from else None
        date_to_dt = parse_date(date_to) if date_to else None
        
        def in_date_range(row_date: str) -> bool:
            dt = parse_date(row_date)
            if dt is None:
                return True  # Include if we can't parse
            if date_from_dt and dt < date_from_dt:
                return False
            if date_to_dt and dt > date_to_dt:
                return False
            return True
        
        mask = df_res["response_date"].apply(in_date_range)
        df_res = df_res[mask]
    
    return df_res


def get_available_states() -> List[str]:
    """Get sorted list of unique states in the dataset."""
    if df is None:
        return []
    states = df["state"].str.upper().str.strip().unique()
    states = [s for s in states if s]  # Remove empty
    return sorted(states)

def format_date(date_str: str) -> str:
    """Format date string to readable format like 'Dec 1, 2026'."""
    if not date_str:
        return ""
    dt = parse_date(date_str)
    if dt:
        return dt.strftime("%b %d, %Y")
    return date_str  # Return original if can't parse

def format_results(df_res: pd.DataFrame) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for _, row in df_res.iterrows():
        title = (row.get("title") or "").strip() or "(no title)"

        agency = (row.get("organization_name") or "").strip()
        if not agency:
            agency = (row.get("full_parent_path_name") or "").strip()

        deadline = format_date((row.get("response_date") or "").strip())

        url = ""
        for col in ("source_url", "ui_link", "additional_info_link"):
            val = row.get(col)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break

        text = (
            row.get("description_text")
            or row.get("combined_text")
            or ""
        )
        text = str(text).replace("\n", " ")
        snippet = text[:400]

        score_val = float(row.get("score", 0.0) or 0.0)
        
        # Include component scores for hybrid mode
        result_dict = {
            "id": row.get("id"),
            "title": title,
            "agency": agency,
            "deadline": deadline,
            "url": url,
            "snippet": snippet,
            "score": f"{score_val:.4f}",
            "state": (row.get("state") or "").strip(),
        }
        
        # Add component scores if available (hybrid mode)
        if "tfidf_score" in row:
            result_dict["tfidf_score"] = f"{row['tfidf_score']:.4f}"
            result_dict["semantic_score"] = f"{row['semantic_score']:.4f}"
        
        results.append(result_dict)
    return results


@app.on_event("startup")
def on_startup():
    init_indexes()


@app.get("/", response_class=HTMLResponse)
async def search_page(
    request: Request,
    q: str = "",
    mode: str = "hybrid",
    page: int = 1,
    per_page: int = 10,
    state: str = "",
    date_from: str = "",
    date_to: str = "",
):
    q = (q or "").strip()
    mode = (mode or "hybrid").lower()
    if mode not in ("tfidf", "semantic", "hybrid"):
        mode = "hybrid"
    page = max(page, 1)
    per_page = max(per_page, 1)

    results: List[Dict[str, Any]] = []
    total = 0
    total_pages = 0
    available_states = get_available_states()

    if q:
        # Run search
        df_res = run_search(q, mode, top_k=100)
        
        # Apply filters
        df_res = apply_filters(df_res, state=state, date_from=date_from, date_to=date_to)

        all_results = format_results(df_res)
        total = len(all_results)
        total_pages = (total + per_page - 1) // per_page

        start = (page - 1) * per_page
        end = start + per_page
        results = all_results[start:end]

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "query": q,
            "mode": mode,
            "results": results,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "state": state,
            "date_from": date_from,
            "date_to": date_to,
            "available_states": available_states,
        },
    )


@app.post("/refresh-local", response_class=HTMLResponse)
async def refresh_local():
    """
    Reload rfp.csv and rebuild TF-IDF + semantic indexes.
    Run ingest_sam.py again before pressing this if you want fresh data.
    """
    init_indexes()
    return RedirectResponse("/", status_code=303)