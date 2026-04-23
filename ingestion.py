"""Extraction et chunking sémantique des documents."""

from __future__ import annotations

import subprocess
from pathlib import Path

from projet_ollama.models import DocumentChunk


def extract_text(source: Path) -> list[tuple[int | None, str]]:
    """Extrait le texte d'un fichier PDF ou texte.

    :param source: Chemin du document à traiter.
    :returns: Une liste de tuples `(page, texte)`.
    :raises FileNotFoundError: Si le fichier source n'existe pas.
    :raises ValueError: Si le format de fichier n'est pas pris en charge.
    :raises RuntimeError: Si `pdftotext` échoue.
    """

    if not source.exists():
        raise FileNotFoundError(f"Document introuvable: {source}")

    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(source)
    if suffix in {".txt", ".md"}:
        return [(None, source.read_text(encoding="utf-8"))]
    raise ValueError(f"Format non pris en charge: {source.suffix}")


def chunk_semantic(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.75,
    min_chunk_sentences: int = 2,
) -> list[str]:
    """Découpe un texte en chunks selon la similarité sémantique entre phrases.

    Regroupe les phrases tant que leur similarité cosinus dépasse le seuil.
    Coupe quand le thème change.

    :param text: Texte à découper.
    :param model: Modèle SentenceTransformer pour encoder les phrases.
    :param similarity_threshold: Seuil en dessous duquel on coupe.
    :param min_chunk_sentences: Nombre minimal de phrases avant une coupure.
    :returns: La liste des chunks produits.
    """

    import nltk

    sentences = nltk.sent_tokenize(text, language="french")
    if len(sentences) < 2:
        return [text] if text.strip() else []

    passages = [f"passage: {s}" for s in sentences]
    embeddings = model.encode(passages, show_progress_bar=False)

    import numpy as np

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        sim /= np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        similarities.append(float(sim))

    chunks: list[str] = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold and len(current_chunk) >= min_chunk_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_document_chunks(
    source: Path,
    similarity_threshold: float = 0.75,
    min_chunk_sentences: int = 2,
    chunking_model: str = "intfloat/multilingual-e5-small",
    page_from: int | None = None,
    page_to: int | None = None,
) -> list[DocumentChunk]:
    """Construit les chunks sémantiques d'un document avec métadonnées.

    :param source: Chemin du document à indexer.
    :param similarity_threshold: Seuil de similarité pour couper les chunks.
    :param min_chunk_sentences: Nombre minimal de phrases avant une coupure.
    :param chunking_model: Modèle SentenceTransformer à utiliser pour le chunking.
    :param page_from: Première page à conserver incluse.
    :param page_to: Dernière page à conserver incluse.
    :returns: Les chunks prêts à être embeddés.
    :raises FileNotFoundError: Si le document n'existe pas.
    :raises ValueError: Si le format n'est pas pris en charge.
    :raises RuntimeError: Si l'extraction échoue.
    """

    import nltk
    from sentence_transformers import SentenceTransformer

    nltk.download("punkt_tab", quiet=True)
    model = SentenceTransformer(chunking_model)

    pages = extract_text(source)
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page, content in pages:
        if not _should_keep_page(page, page_from, page_to):
            continue
        for text in chunk_semantic(content, model, similarity_threshold, min_chunk_sentences):
            page_label = page if page is not None else "na"
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source}:{page_label}:{chunk_index}",
                    source=str(source),
                    text=text,
                    page=page,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return chunks


def _should_keep_page(page: int | None, page_from: int | None, page_to: int | None) -> bool:
    """Indique si une page doit être conservée pour l'indexation.

    :param page: Numéro de page courant.
    :param page_from: Première page autorisée incluse.
    :param page_to: Dernière page autorisée incluse.
    :returns: `True` si la page doit être indexée.
    """

    if page is None:
        return True
    if page_from is not None and page < page_from:
        return False
    if page_to is not None and page > page_to:
        return False
    return True


def _extract_pdf_text(source: Path) -> list[tuple[int | None, str]]:
    """Extrait le texte d'un PDF via `pdftotext`.

    :param source: Chemin du PDF à traiter.
    :returns: Une liste de couples `(numéro de page, texte)`.
    :raises RuntimeError: Si `pdftotext` échoue.
    """

    result = subprocess.run(
        ["pdftotext", str(source), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Échec de l'extraction PDF.")

    pages: list[tuple[int | None, str]] = []
    for index, raw_page in enumerate(result.stdout.split("\f"), start=1):
        page_text = raw_page.strip()
        if page_text:
            pages.append((index, page_text))
    return pages
