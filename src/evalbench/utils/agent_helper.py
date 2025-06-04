import re
import json
from typing import List
from collections import defaultdict
from evalbench.runtime_setup.runtime import get_config

cfg = get_config()

# make confident decisions so downstream processes are deterministic and manageable
def get_user_intent(instruction):
    try:
        prompt = f'''
        You are an intent classification assistant.

        Your task is to determine the user's intent based on their instruction for using the evaluation library.

        Classify the instruction into **one and only one** of the following categories, and return your answer using exactly one of these strings (case-sensitive, no extra text):

        - full_evaluation
        - interpretation_only
        - recommendation_only
        - interpretation and recommendation
        - vague/unclear

        Guidelines:
        - **full_evaluation** → if the user wants to evaluate model outputs using metrics, wants an explanation of the metric results and finally a recommendation to improve based on the result and data.
        - **interpretation_only** → if the user has already run metrics and now wants an explanation or analysis of the scores.
        - **recommendation_only** → if the user wants suggestions on how to improve based on the evaluation results or input behavior.
        - **interpretation and recommendation** → if the user wants both analysis of the scores and suggestions for improvement.
        - **vague/unclear** → if the instruction is incomplete, ambiguous, or doesn't clearly indicate what they want.

        Respond with the intent **only**, nothing else.

        User Instruction:
        \'\'\'{instruction}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        intent = response.choices[0].message['content'].strip()
    except Exception as e:
        intent = None

    return intent

def get_task(instruction, data):
    try:
        prompt = f'''
        You are a task identification assistant.

        Your job is to understand what kind of language task the user is performing, based on their instruction and data. Output a short, descriptive phrase that best captures the nature of the task (e.g., 'summarization', 'retrieval-based question answering', 'chatbot response generation', 'information extraction').

        Be concise. Use only one phrase. Avoid generic labels like 'NLP task'.

        User Instruction:
        \'\'\'{instruction}\'\'\'

        Input Data (if any):
        \'\'\'{data if data else 'N/A'}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=5,
        )
        task = response.choices[0].message['content'].strip()
    except Exception as e:
        task = None

    return task


def parse_data(data):
    json_candidates = re.findall(r'(\{.*?}|\[.*?])', data, re.DOTALL)

    for blob in json_candidates:
        try:
            parsed = json.loads(blob)

            # batch mode
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                keys = set(parsed[0].keys())
                if not all(set(item.keys()) == keys for item in parsed):
                    raise ValueError('Inconsistent keys across batch examples.')

                input_data = defaultdict(list)
                for item in parsed:
                    for key in keys:
                        input_data[key].append(item.get(key, ''))

                return input_data

            # single example mode
            elif isinstance(parsed, dict):
                return parsed

            else:
                raise ValueError(
                    'Missing input data / Unable to extract valid input data. Please ensure your input is in the correct JSON format with required fields.')

        except json.JSONDecodeError:
            break

    return {}

def convert_type(raw_type, expected_type):
    if expected_type == str:
        if isinstance(raw_type, list):
            return ' '.join(map(str, raw_type))
        return str(raw_type)

    elif expected_type == List[str]:
        if isinstance(raw_type, str):
            return [raw_type]
        if isinstance(raw_type, list):
            return [str(v) for v in raw_type]
        return [str(raw_type)]

    elif expected_type == List[List[str]]:
        if isinstance(raw_type, list):
            if all(isinstance(v, str) for v in raw_type):
                return [raw_type]
            if all(isinstance(v, list) for v in raw_type):
                return [[str(i) for i in v] for v in raw_type]
        return [[str(raw_type)]]

    return [str(raw_type)]

def prepare_metric_inputs(validated_metrics, data):
    prepared_metric_inputs = {}
    for metric in validated_metrics:
        metric_info = cfg.metric_registry.get(metric)
        if not metric_info:
            continue

        required_args = metric_info.get('required_args', [])
        arg_types = metric_info.get('arg_types', [])

        metric_inputs = {}
        for arg_name, expected_type in zip(required_args, arg_types):
            raw_value = data.get(arg_name)
            if raw_value is None:
                raise ValueError(f'Missing required input \'{arg_name}\' for metric \'{metric}\'')

            converted_value = convert_type(raw_value, expected_type)
            metric_inputs[arg_name] = converted_value

        prepared_metric_inputs[metric] = metric_inputs

    return prepare_metric_inputs

def generate_report(request):
    instruction = request.get('instruction')
    task = request.get('task', 'Unknown')
    results = request.get('results')
    interpretation = request.get('interpretation')
    recommendations = request.get('recommendations')

    report = f'''
    # LLM Evaluation Report
    
    ## Instruction
    {instruction}
    
    ## Inferred Task
    **{task}**
    '''

    if results:
        report += '## Evaluation Results\n'
        for metric, score in results.items():
            report += f'- **{metric}**: {score}\n'
        report += '\n'

    if interpretation:
        report += f'## Interpretation\n{interpretation.strip()}\n\n'

    if recommendations:
        report += f'## Recommendations\n{recommendations.strip()}\n\n'

    report += '---\n_This report was automatically generated by EvalBench._'

    return report

