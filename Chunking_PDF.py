import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

def chunk_semantic(
    text: str,
    model_name: str = "intfloat/multilingual-e5-small",
    similarity_threshold: float = 0.75,
    min_chunk_sentences: int = 2
) -> list[str]:
    """Chunking sémantique basé sur les embeddings.

    Regroupe les phrases tant que leur similarité dépasse
    le seuil. Coupe quand le thème change.
    """
    # Charger le modèle
    model = SentenceTransformer(model_name)

    # Tokeniser en phrases
    sentences = sent_tokenize(text, language='french')
    if len(sentences) < 2:
        return [text]

    # Encoder toutes les phrases (avec préfixe passage: pour E5)
    passages = [f"passage: {s}" for s in sentences]
    embeddings = model.encode(passages, show_progress_bar=False)

    # Calculer la similarité entre phrases consécutives
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1])
        sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        similarities.append(sim)

    # Créer les chunks aux points de faible similarité
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold and len(current_chunk) >= min_chunk_sentences:
            # Changement de thème détecté
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks






fichier_scraping = "Essential-GraphRAG_texte_final.txt"

with open(fichier_scraping, 'r') as file:
    fichier = file.read()

# --- NOUVELLES LIGNES À REMPLACER/AJOUTER ICI ---

# 1. On exécute la fonction et on sauvegarde la liste des chunks
chunks_generes = chunk_semantic(fichier)

# 2. On affiche un résumé global
print("\n" + "="*50)
print("📊 RÉSUMÉ DU CHUNKING SÉMANTIQUE")
print("="*50)
print(f"Nombre total de chunks créés : {len(chunks_generes)}")
print(f"Taille du texte original     : {len(fichier)} caractères\n")

# 3. On affiche le détail de chaque chunk généré
for i, chunk in enumerate(chunks_generes):
    mots = len(chunk.split())
    
    print(f"--- CHUNK {i + 1} ---")
    print(f"📏 Métriques : {len(chunk)} caractères | {mots} mots")
#    print(f"📝 Contenu   :\n{chunk}")
#    print("-" * 50 + "\n")


#Revoie tous les chunks dans le terminale. 