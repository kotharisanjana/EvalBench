from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.tokenize import word_tokenize
import bert_score as bert
from sentence_transformers import util
from utils.helper import get_config, handle_output, register_metric
import error_handling.validation_helpers as validation

cfg = get_config()

@register_metric('bleu', required_args=['reference', 'generated'], module='generation')
@handle_output()
def bleu_score(reference: List[str], generated: List[str]) -> List[float]:
    """
    :param reference: list of reference strings
    :param generated: list of generated strings
    :return: list of bleu scores
    """
    validation.validate_batch_inputs(reference, generated)
    return [
        sentence_bleu([word_tokenize(ref)], word_tokenize(gen))
        for ref, gen in zip(reference, generated)
    ]

@register_metric('rouge', required_args=['reference', 'generated'], module='generation')
@handle_output()
def rouge_score(reference: List[str], generated: List[str]) -> List[Dict[str, float]]:
    """
    :param reference: list of reference strings
    :param generated: list of generated strings
    :return: list of rouge scores
    """
    validation.validate_batch_inputs(reference, generated)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return [
        {k: v.fmeasure for k, v in scorer.score(ref, gen).items()}
        for ref, gen in zip(reference, generated)
    ]

@register_metric('meteor', required_args=['reference', 'generated'], module='generation')
@handle_output()
def meteor_score(reference: List[str], generated: List[str]) -> List[float]:
    """
    :param reference: list of reference strings
    :param generated: list of generated strings
    :return: list of meteor scores
    """
    validation.validate_batch_inputs(reference, generated)
    return [
        meteor([word_tokenize(ref)], word_tokenize(gen))
        for ref, gen in zip(reference, generated)
    ]

@register_metric('semantic_similarity', required_args=['reference', 'generated'], module='generation')
@handle_output()
def semantic_similarity_score(reference: List[str], generated: List[str]) -> List[float]:
    """
    :param reference: list of reference strings
    :param generated: list of generated strings
    :return: list of semantic similarity scores
    """
    validation.validate_batch_inputs(reference, generated)
    return [
        util.pytorch_cos_sim(
            cfg.sentence_model.encode(ref, convert_to_tensor=True),
            cfg.sentence_model.encode(gen, convert_to_tensor=True)
        ).item()
        for ref, gen in zip(reference, generated)
    ]

@register_metric('bert', required_args=['reference', 'generated'], module='generation')
@handle_output()
def bert_score(reference: List[str], generated: List[str]) -> List[Dict[str, float]]:
    """
    :param reference: list of reference strings
    :param generated: list of generated strings
    :return: list of BERT scores
    """
    validation.validate_batch_inputs(reference, generated)
    precision, recall, f1 = bert.score(generated, reference, lang='en', verbose=False)
    return [
        {
            'precision': precision[i].item(),
            'recall': recall[i].item(),
            'f1': f1[i].item()
        }
        for i in range(len(reference))
    ]