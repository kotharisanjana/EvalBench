from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
from sentence_transformers import SentenceTransformer, util

def bleu(reference, generated):
    bleu_score = sentence_bleu([reference], generated)
    return bleu_score

def rouge(reference, generated):
    scores = rouge_scorer.score(reference, generated)
    return {
        'rouge1': scores['rouge1'],
        'rouge2': scores['rouge2'],
        'rougeL': scores['rougeL']
    }

def meteor(reference, generated):
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    score = meteor_score([reference_tokens], generated_tokens)
    return score

def semantic_similarity(query, answer, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_emb = model.encode(query, convert_to_tensor=True)
    answer_emb = model.encode(answer, convert_to_tensor=True)
    return util.pytorch_cos_sim(query_emb, answer_emb).item()

def bert(reference, generated):
    precision, recall, f1 = bert_score.score([generated], [reference], lang='en', verbose=True)
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }