import numpy as np
from error_handling.validation_helpers import (
    validate_positive_integer,
    validate_list_type_and_non_empty,
    validate_string_non_empty
)
from evalbench_metrics.config import groq_client
from utils.decorators import handle_output, register_metric

@register_metric('recall@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], category='retrieval_search')
@handle_output()
def recall_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    validate_list_type_and_non_empty(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validate_positive_integer(k, 'k')

    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / len(relevant_docs) if relevant_docs else 0.0

@register_metric('precision@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], category='retrieval_search')
@handle_output()
def precision_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    validate_list_type_and_non_empty(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validate_positive_integer(k, 'k')

    relevant_at_k = set(retrieved_docs[:k]).intersection(set(relevant_docs))
    return len(relevant_at_k) / k

@handle_output()
def dcg(relevance_scores: list) -> int:
    return sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(relevance_scores)
    ])

@register_metric('rndcg@k', required_args=['relevant_docs', 'retrieved_docs', 'k'], category='retrieval_search')
@handle_output()
def ndcg_at_k(relevant_docs: list, retrieved_docs: list, k: int) -> float:
    validate_list_type_and_non_empty(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))
    validate_positive_integer(k, 'k')

    rel_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
    ideal_rel_scores = sorted(rel_scores, reverse=True)
    dcg_val = dcg(rel_scores)
    idcg_val = dcg(ideal_rel_scores)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

@register_metric('context_relevance', required_args=['query', 'retrieved_docs'], category='retrieval_search')
@handle_output()
def context_relevance_score(query: str, retrieved_docs: list) -> list:
    validate_string_non_empty(query, 'query')
    validate_list_type_and_non_empty(('retrieved_docs', retrieved_docs))

    scores = []
    for context in retrieved_docs:
        prompt = f'''
        You are evaluating the **relevance** of a retrieved context to a given user query.
        
        A relevant context:
        - Helps answer the query.
        - Is topically and semantically aligned.
        - Avoids being misleading, off-topic, or too vague.
        
        Rate relevance on a scale of **1 to 5**:
        1 = Irrelevant  
        2 = Slightly related  
        3 = Somewhat relevant  
        4 = Mostly relevant  
        5 = Highly relevant and directly useful
        
        ONLY return a single digit (1-5). No explanation.
        
        ---
        
        **Query:** {query}
        
        **Retrieved Context:** {context}
        
        Relevance Score:
        '''

        try:
            response = groq_client.chat.completions.create(
                model='llama3-8b-8192',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=1,
            )
            score = response.choices[0].message.content.strip()
            scores.append(int(score))
        except Exception as e:
            scores.append(0)
    return scores

@register_metric('mrr', required_args=['retrieved_docs', 'relevant_docs'], category='retrieval_search')
@handle_output()
def mrr_score(retrieved_docs: list, relevant_docs: list) -> float:
    validate_list_type_and_non_empty(('relevant_docs', relevant_docs), ('retrieved_docs', retrieved_docs))

    try:
        rank = retrieved_docs.index(relevant_docs) + 1
        return 1 / rank
    except ValueError:
        return 0.0
