import nltk

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

# runtime.py
_active_config = None

def set_config(cfg):
    global _active_config
    _active_config = cfg

def get_config():
    if _active_config is None:
        raise RuntimeError("EvalConfig has not been initialized. Call `set_config(...)` first.")
    return _active_config

