import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Import de votre code local
from vector_store import load_vector_store, search_similar_chunks

# --- Configuration de la page (Doit être en tout premier) ---
st.set_page_config(page_title="Assistant RAG", page_icon="📚")
st.title("Discute avec tes Documents 📚")
st.markdown("Pose tes questions en langage naturel, l'IA cherchera les réponses dans les textes.")

# --- Variables Globales ---
DB_PATH = Path("vector_db.json")
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL = "qwen2.5:1.5b"

# --- Vérification (Hors du cache) ---
if not DB_PATH.exists():
    st.error("❌ Base de données introuvable. Lancez d'abord `python build_db.py`.")
    st.stop()

# 1. Mise en cache de l'initialisation (Identique à app.py)
@st.cache_resource
def init_system():
    # Chargement du modèle de texte (Ollama)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    # Chargement du modèle d'embedding (SentenceTransformer)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    # Connexion à la base de données vectorielle
    vector_store = load_vector_store(DB_PATH)
    
    return embed_model, llm, vector_store

# Tentative de chargement du système
try:
    with st.spinner("Chargement des modèles IA en cours (cela peut prendre quelques secondes)..."):
        embed_model, llm, vector_store = init_system()
except Exception as e:
    st.error(f"❌ Erreur d'initialisation : {e}")
    st.stop()

# 2. Préparation du Prompt LangChain
template = """Tu es un assistant expert.
Voici le contexte extrait de mes documents :
{context}

Écris une réponse précise pour répondre à cette question en te basant UNIQUEMENT sur le contexte : {question}
Si la réponse ne s'y trouve pas, dis-le simplement.
"""
prompt_template = PromptTemplate.from_template(template)
chain = prompt_template | llm

# 3. Gestion de l'historique du chat Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Input utilisateur et traitement
if question := st.chat_input("Ex: Que dit le document sur ce sujet ?"):
    
    # Affichage et sauvegarde de la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Traitement par l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et génération en cours..."):
            
            # Étape A : Calcul du vecteur de la question (préfixe e5)
            query_embedding = embed_model.encode(f"query: {question}", show_progress_bar=False).tolist()
            
            # Étape B : Recherche vectorielle
            resultats = search_similar_chunks(vector_store, query_embedding, limit=3)
            
            # Étape C : Création du contexte
            context_text = "\n\n".join([f"- Extrait (Page {res.chunk.page}) : {res.chunk.text}" for res in resultats])
            
            # Étape D : Génération de la réponse via Ollama
            reponse = chain.invoke({"context": context_text, "question": question})
            
            # Étape E : Formatage visuel pour Streamlit
            st.markdown(reponse.content)
            
            # Affichage des sources rétractables
            with st.expander("📄 Voir les sources utilisées"):
                for i, res in enumerate(resultats, 1):
                    st.markdown(f"**Extrait {i} (Score: {res.score:.2f} | Page {res.chunk.page})**")
                    st.caption(res.chunk.text)
                    
    # Sauvegarde de la réponse dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": reponse.content})