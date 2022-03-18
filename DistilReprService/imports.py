import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
