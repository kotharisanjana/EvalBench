from typing import List
from utils.helper import  get_config, handle_output, register_metric
import error_handling.validation_helpers as validation

@register_metric('faithfulness', required_args=['context', 'generated'], module='contextual_generation')
@handle_output()
def faithfulness_score(context: List[List[str]], generated: List[str]) -> List[float]:
    validation.validate_batch_inputs(context, generated)

    cfg = get_config()
    candidate_labels = ['faithful to context', 'unfaithful to context']
    results = []

    for ctx, gen in zip(context, generated):
        result = cfg.fact_check_model(
            sequences=" ".join(ctx),
            candidate_labels=candidate_labels,
            hypothesis=gen
        )
        labels = result["labels"]
        scores = result["scores"]
        results.append(scores[labels.index("faithful to context")])

    return results

@register_metric('hallucination', required_args=['context', 'generated'], module='contextual_generation')
@handle_output()
def hallucination_score(context: List[List[str]], generated: List[str]) -> List[float]:
    validation.validate_batch_inputs(context, generated)

    cfg = get_config()
    candidate_labels = ["entailment", "neutral", "contradiction"]
    results = []

    for ctx, gen in zip(context, generated):
        result = cfg.fact_check_model(
            sequences=" ".join(ctx),
            candidate_labels=candidate_labels,
            hypothesis=gen
        )
        labels = result["labels"]
        scores = result["scores"]
        # Lower entailment score = higher hallucination likelihood
        entailment_score = scores[labels.index("entailment")]
        results.append(1 - entailment_score)

    return results

@register_metric('groundedness', required_args=['context', 'generated'], module='contextual_generation')
# @handle_output()
def groundedness_score(context: List[List[str]], generated: List[str]) -> List[float]:
    validation.validate_batch_inputs(context, generated)

    cfg = get_config()
    results = []

    for ctx, gen in zip(context, generated):
        prompt = f'''
        You are a helpful evaluator. Given the following retrieved context and the answer, rate how grounded the answer is in the context on a scale of 0 to 5.
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
            results.append(-1.0)

    return results

