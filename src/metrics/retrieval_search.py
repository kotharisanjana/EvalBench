import numpy as np
from typing import List
import error_handling.validation_helpers as validation
from utils.helper import get_config, handle_output, register_metric

cfg = get_config()

@register_metric('recall@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], module='retrieval_search')
@handle_output()
def recall_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    """
    :param relevant_docs: list of relevant documents
    :param retrieved_docs: list of retrieved documents
    :param k: top k
    :return: list of recall@k scores
    """
    validation.validate_batch_inputs(relevant_docs, retrieved_docs)
    validation.validate_type_int_positive_integer(k, 'k')
    return [
        len(set(retrieved[:k]).intersection(set(relevant))) / len(relevant) if len(relevant) > 0 else 0.0
        for retrieved, relevant in zip(retrieved_docs, relevant_docs)
    ]

@register_metric('precision@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], module='retrieval_search')
@handle_output()
def precision_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    """
    :param relevant_docs: list of relevant documents
    :param retrieved_docs: list of retrieved documents
    :param k: top k
    :return: list of precision@k scores
    """
    validation.validate_batch_inputs(relevant_docs, retrieved_docs)
    validation.validate_type_int_positive_integer(k, 'k')
    return [
        len(set(retrieved[:k]).intersection(set(relevant))) / k
        for retrieved, relevant in zip(retrieved_docs, relevant_docs)
    ]

def dcg(relevance_scores: list) -> int:
    return sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

@register_metric('rndcg@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], module='retrieval_search')
@handle_output()
def ndcg_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    """
    :param relevant_docs: list of relevant documents
    :param retrieved_docs: list of retrieved documents
    :param k: top k
    :return: list of ndcg@k scores
    """
    validation.validate_batch_inputs(relevant_docs, retrieved_docs)
    validation.validate_type_int_positive_integer(k, 'k')

    results = []
    for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
        rel_scores = [1 if doc in rel_docs else 0 for doc in ret_docs[:k]]
        ideal_rel_scores = sorted(rel_scores, reverse=True)
        dcg_val = dcg(rel_scores)
        idcg_val = dcg(ideal_rel_scores)
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        results.append(ndcg)

    return results

@register_metric('context_relevance', required_args=['query', 'retrieved_docs'], module='retrieval_search')
@handle_output()
def context_relevance_score(query: List[str], retrieved_docs: List[List[str]]) -> List[float]:
    """
    :param query: list of query
    :param retrieved_docs: list of retrieved docs
    :return: list of context relevance scores
    """
    validation.validate_batch_inputs(query, retrieved_docs)

    results = []
    for q, r in zip(query, retrieved_docs):
        prompt = f'''
        You are a search relevance evaluator. Your task is to score how well a retrieved context matches the user query.
        
        ### Scoring Guidelines:
        1 = Completely irrelevant  
        2 = Slightly related  
        3 = Somewhat relevant  
        4 = Mostly relevant  
        5 = Highly relevant and directly useful for answering the query
        
        ### Instructions:
        - ONLY output the number 1–5. No extra text.
        - Use the full range when appropriate.
        
        ### Examples:
        
        **Query:** "What are the symptoms of heat stroke?"  
        **Context:** "The Eiffel Tower is located in Paris."  
        **Score:** 1
        
        **Query:** "What are the symptoms of heat stroke?"  
        **Context:** "Heat-related illnesses include dehydration, fatigue, and muscle cramps."  
        **Score:** 3
        
        **Query:** "What are the symptoms of heat stroke?"  
        **Context:** "Common symptoms of heat stroke include high body temperature, confusion, rapid pulse, and nausea."  
        **Score:** 5
        
        ### Now rate the following:
        
        **Query:** {q}  
        **Retrieved Context:** {' '.join(r)}  
        
        Relevance Score:
        '''.strip()

        try:
            response = cfg.groq_client.chat.completions.create(
                model='llama3-8b-8192',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=1,
            )
            score = response.choices[0].message.content.strip()
            results.append(float(score))
        except ValueError as e:
            results.append(-1.0)

    return results

@register_metric('mrr', required_args=['retrieved_docs', 'relevant_docs'], module='retrieval_search')
@handle_output()
def mrr_score(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> List[float]:
    """
    :param relevant_docs: list of relevant docs
    :param retrieved_docs: list of retrieved docs
    :param k: top k
    :return: list of mrr scores
    """
    validation.validate_batch_inputs(relevant_docs, retrieved_docs)
    validation.validate_type_int_positive_integer(k, 'k')
    return [
        1/retrieved_docs.index(relevant) + 1
        for relevant in relevant_docs
    ]

