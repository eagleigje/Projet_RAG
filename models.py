"""Modèles métier du projet."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ChatMessage:
    """Représente un message échangé avec le modèle.

    :param role: Rôle du message (`system`, `user` ou `assistant`).
    :param content: Contenu textuel du message.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Sérialise le message pour l'API Ollama.

        :returns: Le message sous forme de dictionnaire.
        """

        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class DocumentChunk:
    """Décrit un fragment de document avant vectorisation.

    :param chunk_id: Identifiant stable du chunk.
    :param source: Chemin source du document.
    :param text: Texte du chunk.
    :param page: Numéro de page d'origine si connu.
    :param chunk_index: Position du chunk dans le document.
    """

    chunk_id: str
    source: str
    text: str
    page: int | None
    chunk_index: int


@dataclass(slots=True)
class StoredChunk:
    """Décrit un chunk accompagné de son embedding.

    :param chunk_id: Identifiant stable du chunk.
    :param source: Chemin source du document.
    :param text: Texte du chunk.
    :param page: Numéro de page d'origine si connu.
    :param chunk_index: Position du chunk dans le document.
    :param embedding: Embedding associé au texte.
    """

    chunk_id: str
    source: str
    text: str
    page: int | None
    chunk_index: int
    embedding: list[float]

    def to_dict(self) -> dict[str, object]:
        """Sérialise le chunk pour le stockage JSON.

        :returns: Une représentation sérialisable en JSON.
        """

        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    """Associe un chunk et un score de similarité.

    :param chunk: Chunk retrouvé dans le vector store.
    :param score: Score de similarité cosinus.
    """

    chunk: StoredChunk
    score: float
