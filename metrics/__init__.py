import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import openai

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
faithfulness_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
factuality_model = pipeline('zero-shot-classification', 'facebook/bart-large-mnli')

openai.api_key = 'your-api-key-here'

