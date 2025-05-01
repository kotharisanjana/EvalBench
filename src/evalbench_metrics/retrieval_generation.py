from sentence_transformers import util
from evalbench_metrics.config import factuality_model, faithfulness_model, sentence_model, groq_client
from utils import handle_output

@handle_output()
def faithfulness_score(context: str, response: str) -> float:
    if not context or not response:
        return float('-inf')
    scores = faithfulness_model.predict([[context, response]])
    return float(scores[0])

@handle_output()
def hallucination_score(context: str, response: str) -> float:
    if not context or not response:
        return float('-inf')
    context_emb = sentence_model.encode(context, convert_to_tensor=True)
    response_emb = sentence_model.encode(response, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(context_emb, response_emb).item()
    return 1 - similarity

@handle_output()
def factuality_score(response: str) -> float:
    if not response:
        return float('-inf')
    result = factuality_model(
        response,
        candidate_labels=['factual', 'non-factual'],
        hypothesis_template='This response is {}.'
    )
    scores = dict(zip(result['labels'], result['scores']))
    return scores.get('factual', 0.0)

@handle_output()
def groundedness_score(context: str, response: str) -> str:
    if not context or not response:
        return ''

    prompt = f'''
    You are a helpful evaluator. Given the following retrieved context and the answer, rate how grounded the answer is in the context on a scale of 1 to 5.
    Context:
    \'\'\'{context}\'\'\'

    Response:
    \'\'\'{response}\'\'\'

    Is the response factual and grounded in the context? Give only the score.
    '''
    completion = groq_client.chat.completions.create(
        model='llama3-8b-8192',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message.content

@handle_output()
def evaluate_all(context: str, response: str) -> dict:
    scores = {}
    if context and response:
        scores['faithfulness_score'] = faithfulness_score(context, response, suppress_output=True)
        scores['hallucination_score'] = hallucination_score(context, response, suppress_output=True)
        scores['factuality_score'] = factuality_score(response, suppress_output=True)
        groundedness = groundedness_score(context, response, suppress_output=True)
        try:
            scores['groundedness_score'] = float(groundedness)
        except ValueError:
            scores['groundedness_score'] = None
    return scores