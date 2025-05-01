from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.tokenize import word_tokenize
import bert_score as bert
from sentence_transformers import util
from evalbench_metrics.config import sentence_model
from utils.decorators import handle_output, register_metric
from error_handling.validation_helpers import (
    validate_string_non_empty
)

@register_metric('bleu', required_args=['reference', 'generated'], category='generation')
@handle_output()
def bleu_score(reference: str, generated: str) -> float:
    validate_string_non_empty(('reference', reference), ('generated', generated))

    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=SmoothingFunction().method4)

@register_metric('rouge', required_args=['reference', 'generated'], category='generation')
@handle_output()
def rouge_score(reference: str, generated: str) -> dict:
    validate_string_non_empty(('reference', reference), ('generated', generated))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in scores.items()}

@register_metric('meteor', required_args=['reference', 'generated'], category='generation')
@handle_output()
def meteor_score(reference: str, generated: str) -> float:
    validate_string_non_empty(('reference', reference), ('generated', generated))

    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    return meteor([reference_tokens], generated_tokens)

@register_metric('semantic_similarity', required_args=['reference', 'generated'], category='generation')
@handle_output()
def semantic_similarity_score(reference: str, generated: str) -> float:
    validate_string_non_empty(('reference', reference), ('generated', generated))

    ref_emb = sentence_model.encode(reference, convert_to_tensor=True)
    gen_emb = sentence_model.encode(generated, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_emb, gen_emb).item()

@register_metric('bert', required_args=['reference', 'generated'], category='generation')
@handle_output()
def bert_score(reference: str, generated: str) -> dict:
    validate_string_non_empty(('reference', reference), ('generated', generated))

    precision, recall, f1 = bert.score(
        [generated], [reference], lang='en', verbose=False,
    )
    return {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item()
    }