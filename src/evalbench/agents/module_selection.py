import ast
import evalbench
from evalbench.runtime_setup.runtime import get_config
from evalbench.utils.agent_helper import prepare_metric_inputs, retry_with_backoff

class ModuleSelection:
    def __init__(self, parsed_request):
        self.cfg = get_config()
        self.parsed_request = parsed_request
        self.available_metrics = list(evalbench.metric_registry.keys())
        self.validated_metrics = []

    def determine_evaluation_metrics(self):
        available_metrics_str = ', '.join(self.available_metrics)

        few_shot_examples = '''
        Example 1:
        User query: 'Evaluate the generated answers using BLEU and ROUGE scores.'
        Response: ['bleu_score', 'rouge_score']
    
        Example 2:
        User query: 'I want to evaluate how well my retrieval system does — maybe precision and recall?'
        Response: ['precision_at_k', 'recall_at_k']
    
        Example 3:
        User query: 'Check if the chatbot responses are coherent and concise.'
        Response: ['coherence_score', 'conciseness_score']
    
        Example 4:
        User query: 'I’m not sure what metrics to use, just help me evaluate the responses.'
        Response: ['bleu_score', 'rouge_score', 'bert_score', 'factuality_score']
    
        Example 5:
        User query: 'I just want to know how relevant the answers are to the queries.'
        Response: ['answer_relevance_score']
        
        Example 6:
        User query: 'Can you run all available retrieval metrics?'
        Response: ['recall_at_k', 'precision_at_k', 'ndcg_at_k', 'mrr_score']
        
        Example 7:
        User query: 'Are the answers relevant and helpful to user queries?'
        Response: ['answer_relevance_score', 'helpfulness_score']
        '''

        def call():
            prompt = f'''
            You are an natural language evaluation assistant with access to the following evaluation metrics (tools):
            {available_metrics_str}
            
            The user could be evaluating the following tasks: {self.parsed_request['task']}. Use this information to determine which metrics to evaluate based on the task.
        
            Given the user query below, do the following:
            - If the query explicitly mentions any metric names from the above list, return those metric names only.
            - If no explicit metrics are mentioned, infer the evaluation task from the query and return a relevant subset of metric names from the list.
            - If no suitable metrics are found, return an empty list.
        
            Respond ONLY with a Python list of metric names.
            
            {few_shot_examples}
        
            User query:
            \'\'\'{self.parsed_request['instruction']}\'\'\'
            '''

            response = self.cfg.groq_client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.5,
            )

            requested_metrics = response.choices[0].message.content.strip()
            requested_metrics = ast.literal_eval(requested_metrics)
            validated_metrics = [m for m in requested_metrics if m in self.available_metrics]
            return validated_metrics

        self.validated_metrics = retry_with_backoff(call)

    def execute(self):
        self.determine_evaluation_metrics()
        metric_inputs_map = prepare_metric_inputs(self.validated_metrics, self.parsed_request['data'])

        results = {}
        for metric, inputs in metric_inputs_map.items():
            metric_info = evalbench.metric_registry.get(metric)
            func = metric_info.get('func')

            try:
                result = func(**inputs)
                results[metric] = result
            except Exception as e:
                results[metric] = {'error': str(e)}

        return results





