from typing import List
from utils.helper import  get_config, handle_output, register_metric
import error_handling.validation_helpers as validation

cfg = get_config()

@register_metric('faithfulness', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def faithfulness_score(context: List[List[str]], generated: List[str]) -> List[float]:
    """
    :param context: list of context strings
    :param generated: list of generated strings
    :return: list of faithfulness scores
    """
    validation.validate_batch_inputs(context, generated)

    candidate_labels = ["faithful to context", "unfaithful to context"]
    results = []
    for ctx, gen in zip(context, generated):
        hypothesis = f"Is the following response faithful to the context? Context: '{' '.join(ctx)}'. Response: '{gen}'"
        score = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
        results.append(score['scores'][1])

    return results

@register_metric('hallucination', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def hallucination_score(context: List[List[str]], generated: List[str]) -> List[float]:
    """
    :param context: list of context strings
    :param generated: list of generated strings
    :return: list of hallucination scores
    """
    validation.validate_batch_inputs(context, generated)

    candidate_labels = ["consistent with context", "hallucinated"]
    results = []
    for ctx, gen in zip(context, generated):
        hypothesis = f"Does the following response align with the given context? Check for hallucination Context: '{' '.join(ctx)}'. Response: '{gen}'"
        score = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
        results.append(score['scores'][1])

    return results

@register_metric('groundedness', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def groundedness_score(context: List[List[str]], generated: List[str]) -> List[float]:
    """
    :param context: list of context strings
    :param generated: list of generated strings
    :return: list of groundedness scores
    """
    validation.validate_batch_inputs(context, generated)

    results = []
    for ctx, gen in zip(context, generated):
        prompt = f'''
        You are a helpful evaluator. Given the following retrieved context and the answer, rate how grounded the answer is in the context on a scale of 1 to 5.
        Context:
        \'\'\'{ctx}\'\'\'
    
        Response:
        \'\'\'{gen}\'\'\'
    
        Is the response factual and grounded in the context? Give only the score.
        '''

        try:
            completion = cfg.groq_client.chat.completions.create(
                model='llama3-8b-8192',
                messages=[{'role': 'user', 'content': prompt}]
            )
            score = float(completion.choices[0].message.content)
            results.append(score)
        except ValueError:
            pass

    return results

@register_metric('answer_relevance', required_args=['query', 'response'], module='retrieval_generation')
@handle_output()
def answer_relevance_score(query: List[str], response: List[str]) -> List[float]:
    """
    :param query: list of query strings
    :param response: list of response strings
    :return: list of answer relevance scores
    """
    validation.validate_batch_inputs(query, response)

    results = []
    for q, r in zip(query, response):
        prompt = f'''
        You are an expert evaluator. Rate how **relevant** a given response is to a specific question, on a scale from **1 to 5**.
        
        ### Scoring Guidelines:
        1 = Completely irrelevant  
        2 = Weakly related, mostly off-topic  
        3 = Partially relevant, some connection  
        4 = Mostly relevant, minor issues  
        5 = Fully relevant, directly answers the question
        
        ### Instructions:
        - Use the full 1–5 scale.
        - ONLY return the number. Do not include explanations or comments.
        
        ### Examples:
        
        **Question:** "What is the capital of France?"  
        **Response:** "Bananas are a good source of potassium."  
        **Rating:** 1
        
        **Question:** "What is the capital of France?"  
        **Response:** "France is a country in Europe."  
        **Rating:** 3
        
        **Question:** "What is the capital of France?"  
        **Response:** "The capital of France is Paris."  
        **Rating:** 5
        
        ### Now evaluate this:
        
        **Question:** {q}  
        **Response:** {r}  
        
        Relevance Score:
        '''.strip()

        try:
            completion = cfg.groq_client.chat.completions.create(
                model='llama3-8b-8192',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=1,
            )
            score = completion.choices[0].message.content.strip()
            results.append(float(score))
        except ValueError:
            pass

    return results

