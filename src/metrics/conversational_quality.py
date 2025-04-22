import openai

def evaluate_conversational_quality(context: str, response: str, metric_type: str) -> float:
    prompt = f"""
    You are a helpful evaluator. Given the following context and response, rate the response based on {metric_type} on a scale of 1 to 5.
    
    Context:
    '''{context}'''
    
    Response:
    '''{response}'''
    
    How {metric_type} is the response? Give a score from 1 to 5.
    """

    completion = openai.ChatCompletion.create(
        model="mixtral-8x7b-32768",  # "llama3-8b-8192"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    score_str = completion["choices"][0]["message"]["content"].strip()

    try:
        return float(score_str)
    except ValueError:
        print(f"Unexpected output from model: {score_str}")
        return -1.0

def coherence_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'coherence')

def conciseness_score(response: str) -> float:
    return evaluate_conversational_quality('', response, 'conciseness')

def helpfulness_score(context: str, response: str) -> float:
    return evaluate_conversational_quality(context, response, 'helpfulness')