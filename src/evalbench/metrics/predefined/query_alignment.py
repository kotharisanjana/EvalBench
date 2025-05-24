from typing import List
from evalbench.utils.helper import get_config, handle_output, register_metric
import evalbench.error_handling.validation_helpers as validation

@register_metric('context_relevance', required_args=['query', 'context'], module='query_alignment')
@handle_output()
def context_relevance_score(query: List[str], context: List[str]) -> List[float]:
    validation.validate_batch_inputs(('context', context), ('query', query))

    cfg = get_config()
    results = []

    for q, ctx in zip(query, context):
        prompt = f'''
        You are a search relevance evaluator. Your task is to score how well a retrieved context matches the user query.

        Scoring Guidelines:
        1 = Completely irrelevant  
        2 = Slightly related  
        3 = Somewhat relevant  
        4 = Mostly relevant  
        5 = Highly relevant and directly useful for answering the query

        Instructions:
        - ONLY output the number 1–5. No extra text.
        - Use the full range when appropriate.

        Examples:
        Query: 'What are the symptoms of heat stroke?'  
        Context: 'The Eiffel Tower is located in Paris.'  
        Score: 1

        Query: 'What are the symptoms of heat stroke?'  
        Context: 'Heat-related illnesses include dehydration, fatigue, and muscle cramps.'  
        Score: 3

        Query: 'What are the symptoms of heat stroke?'  
        Context: 'Common symptoms of heat stroke include high body temperature, confusion, rapid pulse, and nausea.'  
        Score: 5

        Now rate the following:

        Query: {q}  
        Retrieved Context: {ctx}  

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
            results.append(0)

    return results

@register_metric('answer_relevance', required_args=['query', 'response'], module='query_alignment')
@handle_output()
def answer_relevance_score(query: List[str], response: List[str]) -> List[float]:
    validation.validate_batch_inputs(('response', response), ('query', query))

    cfg = get_config()
    results = []

    for q, r in zip(query, response):
        prompt = f'''
        You are an expert evaluator. Rate how relevant a given response is to a specific question, on a scale from 1 to 5.

        Scoring Guidelines:
        1 = Completely irrelevant  
        2 = Weakly related, mostly off-topic  
        3 = Partially relevant, some connection  
        4 = Mostly relevant, minor issues  
        5 = Fully relevant, directly answers the question

         Instructions:
        - Use the full 1–5 scale.
        - ONLY return the number. Do not include explanations or comments.

        Examples:

        Question: 'What is the capital of France?'  
        Response: 'Bananas are a good source of potassium.'  
        Rating: 1

        Question: 'What is the capital of France?'  
        Response: 'France is a country in Europe.'  
        Rating: 3

        Question: 'What is the capital of France?'  
        Response: 'The capital of France is Paris.'  
        Rating:** 5

        Now evaluate this:
        Question: {q}  
        Response: {r}  

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
            results.append(0)

    return results

@register_metric('answer_helpfulness', required_args=['query', 'response'], module='query_alignment')
@handle_output()
def helpfulness_score(query: List[str], response: List[str]) -> List[float]:
    validation.validate_batch_inputs(('response', response), ('query', query))

    cfg = get_config()
    results = []

    for q, r in zip(query, response):
        prompt = f'''
            You are a helpful and fair evaluator. Your task is to assess the following response based on answer helpfulness using a numeric rating between 1 (poor) and 5 (excellent). Respond with only the number.
    
             Instructions:
            - Use the full scale (1 to 5) when evaluating.
            - Do not include any explanation—just return a single number.
            - Assume you're evaluating as a human would: fair, consistent, and strict.
    
            Examples:
            Query: 'How can I improve my public speaking skills?'
            Response: 'Maybe just try not to be nervous or something.'
            Rating: 2
        
            Query: 'How can I improve my public speaking skills?'
            Response: 'Practice regularly, record yourself to evaluate progress, and consider joining a local speaking group like Toastmasters.'
            Rating: 5
        
            Now rate this:
            Query:
            \'\'\"{q}\"\"\"
        
            Response:
            \"\"\"{r}\"\"\"
        
            Rating:
            '''.strip()

        try:
            completion = cfg.groq_client.chat.completions.create(
                model='llama3-8b-8192',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
            )
            score = completion.choices[0].message.content.strip()
            results.append(float(score))
        except ValueError:
            results.append(0)

    return results