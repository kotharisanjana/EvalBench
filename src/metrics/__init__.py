from dotenv import load_dotenv
import os
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from groq import Groq

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

nltk.download('punkt')
nltk.download('wordnet')

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
faithfulness_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
factuality_model = pipeline('zero-shot-classification', 'facebook/bart-large-mnli')

groq_client = Groq()


