"""
RAG (Retrieval-Augmented Generation) Engine for AutoStream Knowledge Base.

Loads the knowledge base from JSON, chunks it, creates embeddings using
sentence-transformers, and stores them in a FAISS vector store for
efficient similarity search at query time.
"""

import json
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    KNOWLEDGE_BASE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


class RAGEngine:
    """Handles knowledge base ingestion, embedding, and retrieval."""

    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self._initialize()

    def _initialize(self):
        """Load knowledge base, create embeddings, and build vector store."""
        documents = self._load_knowledge_base()
        chunks = self._chunk_documents(documents)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"✅ RAG Engine initialized with {len(chunks)} chunks from knowledge base.")

    def _load_knowledge_base(self) -> List[Document]:
        """Parse knowledge_base.json into LangChain Document objects."""
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)

        documents = []

        # ── Company overview ──────────────────────────────────────────────
        documents.append(Document(
            page_content=f"{kb['company']} — {kb['tagline']}",
            metadata={"source": "company_info"},
        ))

        # ── Plans ─────────────────────────────────────────────────────────
        for plan in kb["plans"]:
            features = ", ".join(plan["features"])
            content = (
                f"Plan: {plan['name']}\n"
                f"Price: {plan['price']}\n"
                f"Features: {features}"
            )
            documents.append(Document(
                page_content=content,
                metadata={"source": "plans", "plan_name": plan["name"]},
            ))

        # ── Policies ──────────────────────────────────────────────────────
        policy_text = "Company Policies:\n" + "\n".join(
            f"- {p}" for p in kb["policies"]
        )
        documents.append(Document(
            page_content=policy_text,
            metadata={"source": "policies"},
        ))

        # ── FAQ ───────────────────────────────────────────────────────────
        for item in kb["faq"]:
            content = f"Q: {item['question']}\nA: {item['answer']}"
            documents.append(Document(
                page_content=content,
                metadata={"source": "faq"},
            ))

        return documents

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(documents)

    def retrieve(self, query: str) -> str:
        """
        Retrieve the most relevant knowledge base chunks for a user query.
        Returns a formatted string of the top-K matching documents.
        """
        if not self.vector_store:
            return "Knowledge base not available."

        results = self.vector_store.similarity_search(query, k=TOP_K_RESULTS)

        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"[{i}] {doc.page_content}")

        return "\n\n".join(context_parts)
