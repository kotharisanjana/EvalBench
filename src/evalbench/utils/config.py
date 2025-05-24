import os
import json
import yaml
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq
from evalbench.utils.helper import download_nltk_data

class EvalConfig:
    def __init__(
        self,
        groq_api_key=None,
        download_nltk=True,
        sentence_model='sentence-transformers/all-MiniLM-L6-v2',
        fact_check_model='facebook/bart-large-mnli',
        output_mode='print',
        json_filepath='evaluation_results.json',
        groq_client=None,
    ):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ API key must be provided via constructor or env variable.")

        os.environ['GROQ_API_KEY'] = self.groq_api_key
        if download_nltk:
            download_nltk_data()

        # Initialize once and store
        self.sentence_model = SentenceTransformer(sentence_model)
        self.fact_check_model = pipeline('zero-shot-classification', fact_check_model)
        self.groq_client = groq_client or Groq()

        self.output_mode = output_mode
        self.json_filepath = json_filepath

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, 'r') as f:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                data = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                data = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .yaml or .json")

        return cls(**data)

    # validate config
    def validate(self):
        errors = []

        if not self.groq_api_key:
            errors.append("Missing GROQ API key.")

        if not isinstance(self.output_mode, str) or self.output_mode not in ('print', 'save'):
            errors.append(f"Invalid output_mode: {self.output_mode}")

        # Optional: check model names are strings
        if not isinstance(self.sentence_model, SentenceTransformer):
            errors.append("sentence_model not initialized correctly.")
        if not callable(getattr(self.fact_check_model, "__call__", None)):
            errors.append("fact_check_model not callable (should be a HuggingFace pipeline).")

        if errors:
            raise ValueError("Invalid configuration:\n" + "\n".join(f" - {e}" for e in errors))