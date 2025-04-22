from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
from sentence_transformers import util
from . import sentence_model

def bleu(reference: str, generated: str) -> float:
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens)

def rouge(reference: str, generated: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in scores.items()}

def meteor(reference: str, generated: str) -> float:
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return meteor_score([reference_tokens], generated_tokens)

def semantic_similarity(reference: str, generated: str) -> float:
    ref_emb = sentence_model.encode(reference, convert_to_tensor=True)
    gen_emb = sentence_model.encode(generated, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_emb, gen_emb).item()

def bert_score_metric(reference: str, generated: str) -> dict:
    precision, recall, f1 = bert_score.score(
        [generated], [reference], lang='en', verbose=False,
    )
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }