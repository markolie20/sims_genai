import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Laad het transformer model voor tekst-embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_urls(urls, prompt=None, chunk_size=500, overlap=100, top_k=1):
    """
    Neemt een lijst met URLs en een optionele prompt.
    Returnt de meest relevante chunks over alle URLs als prompt gegeven is.
    Anders returnt het alle chunks.

    Usage:
    chunks = process_urls(['https://example.com', 'https://example2.com'])
    OR
    best_chunks = process_urls(['https://example.com'], prompt="Wat is dit artikel over?")
    """

    def get_wiki_text(url):
        # Haal de HTML op van de URL
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Verwijder onnodige HTML-elementen zoals scripts, styles en navigatie
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        # Verzamel alle paragrafen met voldoende tekst
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 50)
        return text

    def combine_wiki_texts(urls):
        # Combineer de tekst van alle opgegeven URLs
        combined_text = ""
        for url in urls:
            print(f"Ophalen van: {url}")
            try:
                combined_text += get_wiki_text(url) + "\n\n"
            except Exception as e:
                print(f"Fout bij ophalen van {url}: {e}")
        return combined_text

    # Momenteel wordt alleen alle tekst gecombineerd en teruggegeven
    return combine_wiki_texts(urls)
