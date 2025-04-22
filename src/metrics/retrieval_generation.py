import openai
from sentence_transformers import util
from . import factuality_model, faithfulness_model, sentence_model, client

def faithfulness_score(context: str, response: str) -> float:
    scores = faithfulness_model.predict([[context, response]])
    return float(scores[0])

def hallucination_score(context: str, response: str) -> float:
    context_emb = sentence_model.encode(context, convert_to_tensor=True)
    response_emb = sentence_model.encode(response, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(context_emb, response_emb).item()
    return 1 - similarity

def factuality_score(response: str) -> float:
    result = factuality_model(
        response,
        candidate_labels=['factual', 'non-factual'],
        hypothesis_template='This response is {}.'
    )
    scores = dict(zip(result['labels'], result['scores']))
    return scores.get('factual', 0.0)

def g_eval_groundedness(context: str, response: str) -> str:
    prompt = f'''
    You are a helpful evaluator. Given the following retrieved context and the answer, rate how grounded the answer is in the context on a scale of 1 to 5.
    Context:
    \'\'\'{context}\'\'\'

    Response:
    \'\'\'{response}\'\'\'

    Is the response factual and grounded in the context? Give only the score.
    '''
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message