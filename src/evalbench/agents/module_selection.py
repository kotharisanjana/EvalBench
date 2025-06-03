from typing import List
from evalbench.runtime_setup.runtime import get_config

class ModuleSelection:
    def __init__(self, parsed_request):
        self.cfg = get_config()
        self.parsed_request = parsed_request
        self.available_metrics = list(self.cfg.metric_registry.keys())
        self.validated_metrics = []
        self.prepared_metric_inputs = {}

    def determine_evaluation_metrics(self):
        available_metrics_str = ', '.join(self.available_metrics)

        few_shot_examples = """
        Example 1:
        User query: "Evaluate the generated answers using BLEU and ROUGE scores."
        Response: ['bleu_score', 'rouge_score']

        Example 2:
        User query: "I want to evaluate how well my retrieval system does — maybe precision and recall?"
        Response: ['precision_at_k', 'recall_at_k']

        Example 3:
        User query: "Check if the chatbot responses are coherent and concise."
        Response: ['coherence_score', 'conciseness_score']

        Example 4:
        User query: "I’m not sure what metrics to use, just help me evaluate the responses."
        Response: ['bleu_score', 'rouge_score', 'bert_score', 'factuality_score']

        Example 5:
        User query: "I just want to know how relevant the answers are to the queries."
        Response: ['answer_relevance_score']
        
        Example 6:
        User query: "Can you run all available retrieval metrics?"
        Response: ['recall_at_k', 'precision_at_k', 'ndcg_at_k', 'mrr_score']
        
        Example 7:
        User query: "Are the answers relevant and helpful to user queries?"
        Response: ['answer_relevance_score', 'helpfulness_score']
        """

        prompt = f"""
        You are an natural language evaluation assistant with access to the following evaluation metrics (tools):
        {available_metrics_str}

        Given the user query below, do the following:
        - If the query explicitly mentions any metric names from the above list, return those metric names only.
        - If no explicit metrics are mentioned, infer the evaluation task from the query and return a relevant subset of metric names from the list.
        - If no suitable metrics are found, return an empty list.

        Respond ONLY with a Python list of metric names.
        
        {few_shot_examples}

        User query:
        '''{self.parsed_request['goal']}'''
        """

        response = self.cfg.groq_client.chat.completions.create(
            model=self.cfg.llm,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.5,
        )

        try:
            requested_metrics = eval(response.choices[0].message.content.strip())
        except Exception as e:
            requested_metrics = []

        self.validated_metrics = [m for m in requested_metrics if m in self.available_metrics]

    def convert_type(self, raw_type, expected_type):
        if expected_type == str:
            if isinstance(raw_type, list):
                return " ".join(map(str, raw_type))
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

    def prepare_metric_inputs(self):
        input_data = self.parsed_request.get("data", {}).get("input_data", {})

        for metric in self.validated_metrics:
            metric_info = self.cfg.metric_registry.get(metric)
            if not metric_info:
                continue

            required_args = metric_info.get('required_args', [])
            arg_types = metric_info.get('arg_types', [])

            metric_inputs = {}
            for arg_name, expected_type in zip(required_args, arg_types):
                raw_value = input_data.get(arg_name)
                if raw_value is None:
                    raise ValueError(f"Missing required input '{arg_name}' for metric '{metric}'")

                converted_value = self.convert_type(raw_value, expected_type)
                metric_inputs[arg_name] = converted_value

            self.prepared_metric_inputs[metric] = metric_inputs

    def execute(self):
        self.determine_evaluation_metrics()
        self.prepare_metric_inputs()

        results = {}

        for metric in self.validated_metrics:
            metric_info = self.cfg.metric_registry.get(metric)
            if not metric_info:
                continue

            func = metric_info.get('func')
            inputs = self.prepared_metric_inputs.get(metric)
            try:
                result = func(**inputs)
                results[metric] = result
            except Exception as e:
                results[metric] = {"error": str(e)}

        return results






