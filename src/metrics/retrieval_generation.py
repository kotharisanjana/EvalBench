from utils.helper import  get_config
from utils.decorators import handle_output, register_metric
from error_handling.validation_helpers import (
    validate_type_string_non_empty,
    validate_num_args
)

cfg = get_config()

@register_metric('faithfulness', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def faithfulness_score(context: str, generated: str) -> float:
    validate_num_args((('context', context), ('generated', generated)), length=2)
    validate_type_string_non_empty(('context', context), ('generated', generated))

    candidate_labels = ["faithful to context", "unfaithful to context"]
    hypothesis = f"Is the following response faithful to the context? Context: '{context}'. Response: '{generated}'"
    result = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
    score = result['scores'][1]
    return score

@register_metric('hallucination', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def hallucination_score(context: str, generated: str) -> float:
    validate_num_args((('context', context), ('generated', generated)), length=2)
    validate_type_string_non_empty(('context', context), ('generated', generated))

    candidate_labels = ["consistent with context", "hallucinated"]
    hypothesis = f"Does the following response align with the given context? Check for hallucination Context: '{context}'. Response: '{generated}'"
    result = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
    score = result['scores'][1]
    return score

@register_metric('factuality', required_args=['generated'], module='retrieval_generation')
@handle_output()
def factuality_score(generated: str) -> float:
    validate_num_args(('generated', generated), length=1)
    validate_type_string_non_empty(('generated', generated))

    candidate_labels = ["factually correct", "factually incorrect"]
    hypothesis = f"Is the following response factually correct. Response: '{generated}'"
    result = cfg.fact_check_model(generated, candidate_labels, hypothesis=hypothesis)
    score = result['scores'][1]
    return score

@register_metric('groundedness', required_args=['context', 'generated'], module='retrieval_generation')
@handle_output()
def groundedness_score(context: str, generated: str) -> str:
    validate_num_args((('context', context), ('generated', generated)), length=2)
    validate_type_string_non_empty(('context', context), ('generated', generated))

    prompt = f'''
    You are a helpful evaluator. Given the following retrieved context and the answer, rate how grounded the answer is in the context on a scale of 1 to 5.
    Context:
    \'\'\'{context}\'\'\'

    Response:
    \'\'\'{generated}\'\'\'

    Is the response factual and grounded in the context? Give only the score.
    '''
    completion = cfg.groq_client.chat.completions.create(
        model='llama3-8b-8192',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message.content

@register_metric('answer_relevance', required_args=['query', 'response'], module='retrieval_generation')
@handle_output()
def answer_relevance_score(query: str, response: str) -> float:
    validate_num_args((('query', query), ('response', response)), length=2)
    validate_type_string_non_empty(('query', query), ('response', response))

    prompt = f'''
    You are an expert language evaluator. Your task is to assess the **relevance** of a response to a given question.
    
    A relevant response:
    - Directly addresses the question.
    - Provides accurate and specific information.
    - Avoids vague, unrelated, or generic responses.
    
    Instructions:
    - Use a scale from **1 to 5** to rate the relevance:
        1 = Completely irrelevant
        2 = Weakly related, mostly off-topic
        3 = Partially relevant, some connection
        4 = Mostly relevant, minor issues
        5 = Fully relevant and on-topic
    
    ONLY return a single number (1, 2, 3, 4, or 5). Do not explain your reasoning.
    ---
    
    **Question:** {query}

    **Response:** {response}
    
    Relevance Score:
    '''

    response = cfg.groq_client.chat.completions.create(
        model='llama3-8b-8192',
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0,
        max_tokens=1,
    )
    return float(response.choices[0].message.content.strip())