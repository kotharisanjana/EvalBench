from dotenv import load_dotenv
import os
import nltk
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq

# load env
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# download NLTK data if not present
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

# load shared models only if not already cached or available
def load_models():
    download_nltk_data()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    fact_check_model = pipeline('zero-shot-classification', 'facebook/bart-large-mnli')
    return sentence_model, fact_check_model

# Load models
sentence_model, fact_check_model = load_models()

# initialize LLM client
groq_client = Groq()

# output settings
output_mode = 'print'  # default: 'print' (can be 'print' or 'save')
json_filepath = 'evaluation_results.json'