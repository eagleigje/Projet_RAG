import fitz  # PyMuPDF
import re    # Le module pour les expressions régulières

#Configuration
nom_du_pdf = "Essential-GraphRAG.pdf"
nom_fichier_sortie = "Essential-GraphRAG_texte_final.txt"

page_depart_pdf = 22  
page_fin_pdf = 147    

index_depart = page_depart_pdf - 1 
index_fin = page_fin_pdf 

print(f"Ouverture de {nom_du_pdf}...")
document = fitz.open(nom_du_pdf)

texte_complet = ""

numero_vraie_page = 1 

#Lecture et nettoyage par page
for i in range(index_depart, index_fin):

    if i >= len(document): #Sécurité
        break 
        
    page = document[i]
    texte_page = page.get_text("text")
    
    
    texte_page = re.sub(r'(?<!\n)\n(?!\n)', ' ', texte_page) #Supprime les retours à la ligne qui coupent les phrases en 2
    texte_page = re.sub(r'[ \t]+', ' ', texte_page) #Supprime les gros espaces
    

    texte_complet += f"\n\n--- PAGE {numero_vraie_page} ---\n\n"
    
    texte_complet += texte_page.strip()
    
    numero_vraie_page += 1

with open(nom_fichier_sortie, "w", encoding="utf-8") as f:
    f.write(texte_complet.strip())

print(f"Extraction terminée !")
print(f"Les pages {page_depart_pdf} à {page_fin_pdf} du PDF ont été extraites.")
print(f"Elles ont été renumérotées de 1 à {numero_vraie_page - 1}.")
print(f"Le résultat est sauvegardé dans : {nom_fichier_sortie}")