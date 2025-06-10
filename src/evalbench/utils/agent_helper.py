import json
import time
from collections import defaultdict
import evalbench
from evalbench.runtime_setup.runtime import get_config

def plan_steps(instruction):
    cfg = get_config()

    def call():
        prompt = f'''
        You are a planning assistant for an LLM evaluation library called EvalBench.

        Your task is to determine which of the following steps should be executed to fulfill the user's instruction.  
        These steps must be executed in this strict order because they are interdependent:
        
        1. evaluation → Run one or more evaluation metrics on model outputs to produce structured results (e.g., accuracy, relevance, coherence, etc.).
        2. interpretation → Analyze and explain the evaluation results to help the user understand what the scores mean and what they reveal about the model's behavior.
        3. recommendation → Based on the evaluation results (and optionally their interpretation), suggest actions to improve model performance, prompt design, data, or evaluation strategy.
        
        Instructions:
        - Only include a step if the user instruction clearly asks for it (either explicitly or implicitly).  
        - Output a Python list of strings in the correct order, using only the terms: 'evaluation', 'interpretation', 'recommendation'.
        - Return only the list without any additional text or explanation preceeding it or following it.
        - If the instruction is unclear or does not map to any of these steps, return: `[]`
        
        ---
        User Instruction:
        \'\'\'{instruction}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    return retry_with_backoff(call)

def get_task(instruction, data):
    cfg = get_config()

    def call():
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
        return response.choices[0].message.content.strip()

    return retry_with_backoff(call)

def parse_data(steps_to_execute, data):
    if 'evaluation' in steps_to_execute:
        if not data:
            raise ValueError('Data is required for evaluation tasks.')

    if isinstance(data, (dict, list)):
        data_str = json.dumps(data)
    else:
        data_str = data

    try:
        parsed = json.loads(data_str)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON input data.")

    # If parsed is a list of dicts, extract input_data dict of lists
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        keys = set(parsed[0].keys())
        if not all(set(item.keys()) == keys for item in parsed):
            raise ValueError('Inconsistent keys across batch examples.')

        input_data = defaultdict(list)
        for item in parsed:
            for key in keys:
                input_data[key].append(item.get(key, ''))

        return input_data

    else:
        raise ValueError(
            'Missing input data / Unable to extract valid input data. Please ensure your input is a list of dicts with required fields.')


# def convert_type(raw_type, expected_type):
#     if expected_type == str:
#         if isinstance(raw_type, list):
#             return ' '.join(map(str, raw_type))
#         return str(raw_type)
#
#     elif expected_type == List[str]:
#         if isinstance(raw_type, str):
#             return [raw_type]
#         if isinstance(raw_type, list):
#             return [str(v) for v in raw_type]
#         return [str(raw_type)]
#
#     elif expected_type == List[List[str]]:
#         if isinstance(raw_type, list):
#             if all(isinstance(v, str) for v in raw_type):
#                 return [raw_type]
#             if all(isinstance(v, list) for v in raw_type):
#                 return [[str(i) for i in v] for v in raw_type]
#         return [[str(raw_type)]]
#
#     return [str(raw_type)]

def prepare_metric_inputs(validated_metrics, data):
    metric_inputs_map = {}

    for metric in validated_metrics:
        metric_info = evalbench.metric_registry.get(metric)
        if not metric_info:
            continue

        required_args = metric_info.get('required_args', [])
        # arg_types = metric_info.get('arg_types', [])

        if all(k in data for k in required_args):
            metric_inputs_map[metric] = data

    return metric_inputs_map

def improve_prompt(instruction):
    cfg = get_config()

    try:
        prompt = f'''
        You are a prompt improvement assistant for an evaluation library called EvalBench.

        Your goal is to help the user rewrite their instruction to be clearer and more specific, so it can be mapped to one or more of the following steps in an evaluation pipeline:
        - evaluation → Run metrics like accuracy, coherence, relevance, etc. on model outputs.
        - interpretation → Explain what the evaluation scores mean and what they reveal.
        - recommendation → Suggest improvements based on the evaluation results or their interpretation.

        Only rewrite the instruction if it's ambiguous or vague. Avoid changing its meaning unnecessarily. If it's unclear what the user wants, try offering different possible directions.

        Respond with:
        'Sorry, I couldn’t understand your instruction. Here are some improvised ways you could phrase it:'

        Then give 2–3 improved examples that:
        - Are clearer and more actionable
        - Explicitly or implicitly map to one or more of the steps above
        - Use natural, user-friendly language
        
        ---
        Original user instruction:
        \'\'\'{instruction}\'\'\'
        '''

        response = cfg.groq_client.chat.completions.create(
            model=cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.5,
        )

        improved_instruction = response.choices[0].message.content.strip()
    except Exception as e:
        improved_instruction = 'Sorry, unable to provide instruction improvements at this time. Please try rephrasing your request.'

    return improved_instruction

def retry_with_backoff(func, max_retries=3, initial_delay=1, *args, **kwargs):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay = 2
    return None