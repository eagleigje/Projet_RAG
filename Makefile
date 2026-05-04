# Variables pour les exécutables
PYTHON = python3
STREAMLIT = streamlit

# Noms des fichiers générés
TXT_FILE = Essential-GraphRAG_texte_final.txt
DB_FILE = vector_db.json

# Cible par défaut (affiche l'aide)
.PHONY: help
help:
	@echo "--- Commandes du Projet RAG ---"
	@echo "make install   : Installe les dépendances Python nécessaires"
	@echo "make data      : Étape 1 - Extrait le texte du PDF (scraping)"
	@echo "make index     : Étape 2 - Crée la base vectorielle JSON (build_db)"
	@echo "make run       : Étape 3 - Lance l'interface de chat Streamlit"
	@echo "make full      : Exécute TOUT le pipeline (Data + Index + Run)"
	@echo "make clean     : Supprime les fichiers générés (.txt, .json) et les caches"

# Installation des dépendances
.PHONY: install
install:
	pip install streamlit sentence-transformers langchain-ollama pymupdf nltk

# Extraction du texte (Génère le .txt)
.PHONY: data
data:
	@echo "🔨 Extraction du texte depuis le PDF..."
	$(PYTHON) scraping_pdf.py

# Création de la base vectorielle (Génère le .json)
# Cette cible dépend de la présence du fichier .txt
.PHONY: index
index:
	@if [ ! -f $(TXT_FILE) ]; then $(MAKE) data; fi
	@echo "🧠 Création de la base de données vectorielle..."
	$(PYTHON) build_db.py

# Lancement de l'interface
.PHONY: run
run:
	@if [ ! -f $(DB_FILE) ]; then \
		echo "⚠️ Base de données manquante. Lancement de l'indexation automatique..."; \
		$(MAKE) index; \
	fi
	@echo "🚀 Lancement de Streamlit..."
	$(STREAMLIT) run main.py

# Pipeline complet
.PHONY: full
full: clean data index run

# Nettoyage des fichiers créés
.PHONY: clean
clean:
	@echo "🧹 Nettoyage des fichiers générés..."
	rm -f $(TXT_FILE)
	rm -f $(DB_FILE)
	@echo "🗑️ Suppression des caches Python..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Nettoyage terminé."