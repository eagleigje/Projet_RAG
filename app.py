import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Configuration de la page
st.set_page_config(page_title="Assistant SQL", page_icon="🗄️")
st.title("Discute avec ta Base de Données 🗄️")
st.markdown("Pose tes questions en langage naturel, l'IA générera et exécutera le SQL pour toi.")

load_dotenv(override=True)

# 1. Mise en cache de l'initialisation (TRÈS IMPORTANT pour Streamlit)
@st.cache_resource
def init_system():
    # Chargement du modèle
    model = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
    
    # Connexion BDD
    DB_USER = "postgres"
    DB_PASSWORD = "1234" 
    DB_HOST = "localhost" 
    DB_PORT = "5432" 
    DB_NAME = "postgres" 
    db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    db = SQLDatabase.from_uri(db_uri)
    
    return model, db

# Tentative de chargement du système
try:
    model, db = init_system()
except Exception as e:
    st.error(f"❌ Erreur d'initialisation (Vérifie Ollama et PostgreSQL) : {e}")
    st.stop()

# 2. Préparation du Prompt LangChain
schema_bd = db.get_table_info() 

template = """Tu es un expert PostgreSQL.
Voici le schéma de ma base de données :
{schema}

Écris la requête SQL exacte pour répondre à cette question : {question}
Ne renvoie RIEN D'AUTRE que le code SQL (pas de texte, pas d'explications).
"""
prompt_template = PromptTemplate.from_template(template)
chain = prompt_template | model

# 3. Gestion de l'historique du chat Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Input utilisateur et traitement
if question := st.chat_input("Ex: Affiche les 3 premiers pays de la table region"):
    
    # Affichage et sauvegarde de la question
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Traitement par l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Génération et exécution de la requête en cours..."):
            # Étape A : Génération de la requête
            reponse = chain.invoke({"schema": schema_bd, "question": question})
            sql_query = reponse.content.replace("```sql", "").replace("```", "").strip()
            
            # Étape B : Exécution de la requête
            resultat = db.run(sql_query)
                
            # Étape C : Formatage visuel pour Streamlit
            reponse_visuelle = f"**Requête générée :**\n"


