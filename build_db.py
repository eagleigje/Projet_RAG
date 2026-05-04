import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- Vos modules métier ---
from ingestion import build_document_chunks
from models import StoredChunk
from vector_store import save_vector_store

# --- Configuration ---
DOC_PATH = Path("Essential-GraphRAG_texte_final.txt")
DB_PATH = Path("vector_db.json")
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

def create_database():
    print(f"🚀 Démarrage de l'indexation de la base de données...")
    start_time = time.time()

    # 1. Chargement du modèle d'embedding
    print(f"📦 Chargement du modèle : {EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2. Ingestion et Chunking
    print(f"📄 Lecture et découpage du document : {DOC_PATH}...")
    try:
        chunks = build_document_chunks(
            source=DOC_PATH,
            chunking_model=EMBED_MODEL_NAME
        )
        print(f"✅ {len(chunks)} chunks générés avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du découpage : {e}")
        return

    # 3. Calcul des Embeddings
    print("🧠 Calcul des vecteurs (embeddings) en cours...")
    stored_chunks = []
    for i, c in enumerate(chunks, 1):
        # Le modèle e5 nécessite le préfixe "passage: " pour les documents stockés
        texte_prepare = f"passage: {c.text}"
        embedding = embed_model.encode(texte_prepare, show_progress_bar=False).tolist()
        
        # Création de l'objet prêt à être stocké
        sc = StoredChunk(
            chunk_id=c.chunk_id,
            source=c.source,
            text=c.text,
            page=c.page,
            chunk_index=c.chunk_index,
            embedding=embedding
        )
        stored_chunks.append(sc)
        
        # Petit affichage de progression
        if i % 50 == 0 or i == len(chunks):
            print(f"   -> {i}/{len(chunks)} chunks vectorisés...")

    # 4. Sauvegarde dans le Vector Store
    print(f"💾 Sauvegarde de la base de données vers {DB_PATH}...")
    save_vector_store(DB_PATH, EMBED_MODEL_NAME, stored_chunks)

    elapsed_time = round(time.time() - start_time, 2)
    print(f"🎉 Terminé ! Base de données créée en {elapsed_time} secondes.")

if __name__ == "__main__":
    # Exécuter ce script uniquement si on le lance directement
    create_database()