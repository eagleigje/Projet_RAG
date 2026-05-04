"""Vector store local basé sur un fichier JSON."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from models import SearchResult, StoredChunk


@dataclass(slots=True)
class VectorStoreData:
    """Regroupe le contenu chargé depuis le vector store.

    :param embedding_model: Modèle d'embedding utilisé pour l'index.
    :param chunks: Chunks stockés avec leurs embeddings.
    """

    embedding_model: str
    chunks: list[StoredChunk]


def save_vector_store(
    path: Path,
    embedding_model: str,
    chunks: list[StoredChunk],
) -> None:
    """Sauvegarde le vector store en JSON.

    :param path: Emplacement du fichier à écrire.
    :param embedding_model: Modèle d'embedding utilisé.
    :param chunks: Chunks à persister.
    :returns: `None`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": embedding_model,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vector_store(path: Path) -> VectorStoreData:
    """Charge le vector store depuis le disque.

    :param path: Emplacement du fichier JSON.
    :returns: Le vector store désérialisé.
    :raises FileNotFoundError: Si le fichier n'existe pas.
    :raises ValueError: Si le contenu JSON est invalide.
    """

    if not path.exists():
        raise FileNotFoundError(f"Vector store introuvable: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Le vector store est invalide : champ 'chunks' manquant.")

    chunks = [StoredChunk(**raw_chunk) for raw_chunk in raw_chunks]
    return VectorStoreData(
        embedding_model=str(payload.get("embedding_model", "")),
        chunks=chunks,
    )


def search_similar_chunks(
    store: VectorStoreData,
    query_embedding: list[float],
    limit: int,
) -> list[SearchResult]:
    """Retourne les chunks les plus proches selon la similarité cosinus.

    :param store: Vector store chargé en mémoire.
    :param query_embedding: Embedding de la question.
    :param limit: Nombre maximal de résultats.
    :returns: Les résultats triés par score décroissant.
    :raises ValueError: Si le vector store est vide.
    """

    if not store.chunks:
        raise ValueError("Le vector store est vide.")

    scored_chunks = [
        SearchResult(chunk=chunk, score=_cosine_similarity(query_embedding, chunk.embedding))
        for chunk in store.chunks
    ]
    scored_chunks.sort(key=lambda result: result.score, reverse=True)
    return scored_chunks[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs.

    :param left: Premier vecteur.
    :param right: Second vecteur.
    :returns: Le score de similarité cosinus.
    :raises ValueError: Si les dimensions diffèrent.
    """

    if len(left) != len(right):
        raise ValueError("Les dimensions d'embedding ne correspondent pas.")

    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)
