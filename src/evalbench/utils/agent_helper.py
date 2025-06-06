import re
import json
from typing import List
from collections import defaultdict
import evalbench
from evalbench.runtime_setup.runtime import get_config

# make confident decisions so downstream processes are deterministic and manageable
def get_user_intent(instruction):
    cfg = get_config()
    try:
        prompt = f'''
        You are an intent classification assistant.
        Your task is to determine the user's intent based on their instruction for using the evaluation library.
        Classify the instruction into one and only one of the following categories, and return your answer using exactly one of these strings (case-sensitive, no extra text):
        - full_pipeline
        - evaluation_only
        - interpretation_only
        - recommendation_only
        - interpretation and recommendation
        - vague/unclear
        
        Guidelines:
        - full_pipeline → if the user wants to evaluate model outputs using metrics, wants an explanation of the metric results and finally a recommendation to improve based on the result and data.
        - evaluation_only → if the user wants to run metrics on model outputs without any further analysis or recommendations.
        - interpretation_only → if the user has already run metrics and now wants an explanation or analysis of the scores.
        - recommendation_only → if the user wants suggestions on how to improve based on the evaluation results or input behavior.
        - interpretation and recommendation → if the user wants both analysis of the scores and suggestions for improvement.
        - vague/unclear → if the instruction is incomplete, ambiguous, or doesn't clearly indicate what they want.
        - Respond with the intent only, nothing else.
        
        User Instruction:
        \'\'\'{instruction}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        intent = response.choices[0].message.content.strip()
    except Exception as e:
        intent = None

    return intent

def get_task(instruction, data):
    cfg = get_config()
    try:
        prompt = f'''
        You are a task identification assistant.
        Your job is to identify the underlying NLP task the user is working on based on their instruction and data. 
        The user may talk about evaluation, interpretation, or improvement, but your focus is only on identifying the core language task being performed — such as question answering, summarization, dialogue generation, etc.

        Guidelines:
        - Output only the name of the NLP task.
        - Your answer should be short (3-5 words max), lowercase, and specific (e.g., 'retrieval-based question answering', 'document summarization', 'chatbot response generation').
        - Ignore mentions of evaluation, interpretation, or improvement. Focus on what kind of language task the model is being used for.
        - If the task is unclear or ambiguous, return: `unknown`

        ---

        Examples:

        Instruction:
        'Is the answer factually accurate and relevant to the user’s query?'
        Data:
        {{'query': 'What is photosynthesis?', 'response': 'It is how plants make energy from sunlight.'}}
        → retrieval-based question answering

        Instruction:
        'Check if the summary captures all the key points and suggest improvements.'
        Data:
        {{'text': '...', 'summary': '...'}}
        → document summarization

        Instruction:
        'Evaluate the response quality and help me improve my chatbot.'
        Data:
        {{'query': 'What's the weather today?', 'response': 'Hi there! I'm not sure.'}}
        → chatbot response generation

        Instruction:
        'Evaluate this for coherence.'
        Data:
        N/A
        → unknown

        ---

        Now, identify the task for the input below.

        User Instruction:
        \'\'\'{instruction}\'\'\'

        Input Data:
        \'\'\'{data if data else 'N/A'}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=1,
        )
        task = response.choices[0].message.content.strip()
    except Exception as e:
        task = None

    return task

def parse_data(data):
    if isinstance(data, (dict, list)):
        data = json.dumps(data)

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
    metric_inputs_map = {}

    for metric in validated_metrics:
        metric_info = evalbench.metric_registry.get(metric)
        if not metric_info:
            continue

        required_args = metric_info.get('required_args', [])
        arg_types = metric_info.get('arg_types', [])

        if all(k in data for k in required_args):
            metric_inputs = {}
            for arg_name, expected_type in zip(required_args, arg_types):
                raw_value = data.get(arg_name)

                converted_value = convert_type(raw_value, expected_type)
                metric_inputs[arg_name] = converted_value

            metric_inputs_map[metric] = metric_inputs

    return metric_inputs_map

def generate_report(request):
    instruction = request.get('instruction', 'N/A').strip()
    task = request.get('task', 'Unknown').strip()
    results = request.get('results', {})
    interpretation = request.get('interpretation', '').strip()
    recommendations = request.get('recommendations', '').strip()

    report = ['---', 'LLM Evaluation Report\n', f'Instruction:\n{instruction}\n', f'Inferred Task:\n{task}\n']

    if results:
        report.append('Evaluation Results:')
        for metric, score in results.items():
            if isinstance(score, list) and len(score) == 1:
                score_str = score[0]
            else:
                score_str = str(score)
            report.append(f'- {metric}: {score_str}')
        report.append('')  # spacing

    if interpretation:
        report.append('Interpretation:')
        report.append(interpretation)
        report.append('')

    if recommendations:
        report.append('Recommendations:')
        report.append(recommendations)
        report.append('')

    report.append('---')
    report.append('This report was automatically generated by EvalBench.')

    return '\n'.join(report)

