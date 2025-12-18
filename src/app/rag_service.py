from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _docs_from_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                if "ingredient" in row and "calories" in row:
                    txt = (
                        f"Ingredient: {row.get('ingredient')}. "
                        f"Calories: {row.get('calories')} kcal, Protein: {row.get('protein_g')}g, "
                        f"Carbs: {row.get('carbs_g')}g, Fat: {row.get('fat_g')}g. "
                        f"Allergens: {row.get('allergens')}. Diet tags: {row.get('diet_tags')}."
                    )
                elif "substitute" in row:
                    txt = (
                        f"Substitution: Replace {row.get('ingredient')} with {row.get('substitute')}. "
                        f"Reason: {row.get('reason')}."
                    )
                else:
                    txt = " ".join(f"{k}: {v}" for k, v in row.items())
                docs.append(Document(page_content=txt, metadata={"source": path.name}))
    except Exception:
        return []
    return docs


def _docs_from_excel(path: Path) -> List[Document]:
    """
    Convert each row of an Excel sheet to a Document. Keeps all non-empty columns as key:value pairs.
    """
    docs: List[Document] = []
    try:
        df = pd.read_excel(path, engine="openpyxl")
        df = df.fillna("")
        for idx, row in df.iterrows():
            # Build a compact text block of available fields.
            parts = [f"{col}: {str(val).strip()}" for col, val in row.items() if str(val).strip()]
            if not parts:
                continue
            text = "; ".join(parts)
            docs.append(Document(page_content=text, metadata={"source": path.name, "row": int(idx)}))
    except Exception:
        return []
    return docs


def load_seed_documents(root: Path) -> List[Document]:
    """
    Load markdown/txt files from docs/ and nutrition data from Excel/CSV in data/.
    Safe to call even if files are missing.
    """
    docs: List[Document] = []
    docs_dir = root / "docs"
    data_dir = root / "data"

    for path in docs_dir.glob("*.md"):
        txt = _read_text_file(path)
        if txt:
            docs.append(Document(page_content=txt, metadata={"source": path.name}))
    for path in docs_dir.glob("*.txt"):
        txt = _read_text_file(path)
        if txt:
            docs.append(Document(page_content=txt, metadata={"source": path.name}))
    # Prefer the richer nutrition.xlsx dataset if present.
    nutrition_xlsx = data_dir / "nutrition.xlsx"
    if nutrition_xlsx.exists():
        docs.extend(_docs_from_excel(nutrition_xlsx))
    else:
        # Fallback: ingest all CSVs.
        for path in data_dir.glob("*.csv"):
            docs.extend(_docs_from_csv(path))
    return docs


class RagService:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.store = Chroma(
            collection_name="nutrition",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def is_empty(self) -> bool:
        try:
            return self.store._collection.count() == 0
        except Exception:
            return False

    def ingest_documents(self, docs: Iterable[Document]) -> int:
        split_docs = self.splitter.split_documents(list(docs))
        if not split_docs:
            return 0
        self.store.add_documents(split_docs)
        self.store.persist()
        return len(split_docs)

    def ingest_texts(self, texts: List[str], metadata: Optional[List[dict]] = None) -> int:
        metadata = metadata or [{} for _ in texts]
        docs: List[Document] = []
        for txt, meta in zip(texts, metadata):
            if not txt:
                continue
            docs.append(Document(page_content=txt, metadata=meta or {}))
        return self.ingest_documents(docs)

    def search_context(self, query: str, k: int = 4) -> tuple[str, List[dict]]:
        """
        Retrieve top-k docs and return a context block plus source metadata for prompting.
        """
        hits = self.store.similarity_search(query, k=k)
        context = "\n\n".join([h.page_content for h in hits])
        sources = [dict(h.metadata or {}, text=h.page_content) for h in hits]
        return context, sources

    def build_prompt(self, question: str, k: int = 4) -> tuple[str, List[dict]]:
        hits = self.store.similarity_search(question, k=k)
        context = "\n\n".join([h.page_content for h in hits])
        prompt = (
            "You are a nutrition assistant. Answer the question using only the context.\n"
            "Cite ingredients, diet rules, or substitutions when relevant.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        sources = [dict(h.metadata or {}, text=h.page_content) for h in hits]
        return prompt, sources
