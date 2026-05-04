# Projet_RAG

QUE FAIT LES FICHIER : 

1. Nettoyage du PDF et conversion en texte -> scraping_pdf.py

2. Création de la base de données -> build_db.py
Utilise :  
- ingestion.py (pour découper le texte)  
- models.py (pour structurer la donnée)
- vector_store.py (pour sauvegarder le JSON)

3. interface utilisateur -> main.py
Utilise : 
- vector_store.py (pour charger et chercher dans le JSON)
- models.py (indirectement, pour lire les résultats)


(fichier non utilisé :app.py car pas adapté au fichier JSON et Chunking_pdf.py car fait la même chose que ingestion.pdf) 

COMMENT FAIRE TOURNER LE PROJET : 

ensemble des bibliothèque à installer :
streamlit sentence-transformers langchain-ollama pymupdf nltk

utiliser le Makefile : 
make run 
ça lance le projet et créer la base de données si besoin

make clean 
supprime les fichiers adjacent genre base de données fichier txt etc

make data index
force la mise à jour des données




créer la base de données si ce n'est pas déja fait : 

exécuter les fichiers :

scraping_pdf.py
build_db.py


lancer ollama dans le terminal et le laisser en arrière plan : 
ollama run qwen2.5:1.5b

lancer streamlit dans le terminal : 
streamlit run main.py

