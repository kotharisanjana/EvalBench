from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.tokenize import word_tokenize
import bert_score as bert
from sentence_transformers import util
from .config import sentence_model

def bleu_score(reference: str, generated: str) -> float:
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens)

def rouge_score(reference: str, generated: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in scores.items()}

def meteor_score(reference: str, generated: str) -> float:
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return meteor([reference_tokens], generated_tokens)

def semantic_similarity_score(reference: str, generated: str) -> float:
    ref_emb = sentence_model.encode(reference, convert_to_tensor=True)
    gen_emb = sentence_model.encode(generated, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_emb, gen_emb).item()

def bert_score(reference: str, generated: str) -> dict:
    precision, recall, f1 = bert.score(
        [generated], [reference], lang='en', verbose=False,
    )
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }

def evaluate_all(reference: str, generated: str) -> dict:
    scores = {}
    if reference and generated:
        scores['bleu_score'] = bleu_score(reference, generated)
        scores['rouge_score'] = rouge_score(reference, generated)
        scores['meteor_score'] = meteor_score(reference, generated)
        scores['semantic_similarity_score'] = semantic_similarity_score(reference, generated)
        scores['bert_score'] = bert_score(reference, generated)
    return scores