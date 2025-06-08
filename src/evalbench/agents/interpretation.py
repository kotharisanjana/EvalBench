from evalbench.runtime_setup.runtime import get_config
from evalbench.utils.agent_helper import retry_with_backoff


class Interpretation:
    def __init__(self, parsed_request):
        self.cfg = get_config()
        self.parsed_request = parsed_request

    def interpret(self, results=None):
        metric_results = self.parsed_request['results'] if self.parsed_request['results'] else results

        def call():
            prompt = f'''
            You are an expert model evaluation analyst.
    
            Your task is to interpret a set of evaluation metric results computed over multiple model outputs. Based on these scores, provide a concise, holistic analysis of the model’s overall performance, strengths, and weaknesses.
            
            Instructions:
            - Analyze the aggregate patterns across all metrics together, not just metric-by-metric.
            - Provide an integrated summary that synthesizes what the numbers reveal about the system’s behavior.
            - Highlight key tradeoffs or tensions between metrics.
            - Avoid listing or explaining each metric separately.
            - Use concrete, data-driven reasoning tied to the task.
            - Focus on the whole system’s quality, not on individual data points.
            - Return a numbered list of 2–3 concise, insightful points capturing the overall picture.
            - Do not include recommendations or next steps; focus only on interpretation.
            
            Example:
            Metric Results:
            {{
              'bleu_score': [0.21, 0.25, 0.22],
              'rouge_score': [0.30, 0.33, 0.31],
              'bert_score': [0.78, 0.80, 0.77]
            }}
            Interpretation:
            1. The model maintains strong semantic relevance overall (high BERT scores), indicating good meaning preservation.
            2. Lexical overlap metrics (BLEU and ROUGE) are lower, suggesting that while the meaning is retained, surface phrasing varies substantially.
            3. This pattern implies the model is effective at generating diverse but semantically aligned responses, which may be advantageous in creative or flexible language generation tasks.
    
            Example 2:
            Metric Results:
            {{
              'recall_at_k': [0.92, 0.95, 0.90],
              'precision_at_k': [0.45, 0.50, 0.48],
              'mrr_score': [0.35, 0.40, 0.38]
            }}
            Interpretation:
            The system retrieves most of the relevant documents (high recall), but many of the retrieved results are irrelevant (low precision). Additionally, relevant documents often appear lower in the list (low MRR), indicating room for improvement in ranking. Consider improving ranking heuristics or embedding quality.
            
            Example 3:
            Metric Results:
            {{
              'helpfulness_score': [0.55, 0.48, 0.52],
              'coherence_score': [0.82, 0.85, 0.80],
              'factuality_score': [0.65, 0.62, 0.60]
            }}
            Interpretation:
            The chatbot's responses are mostly coherent and factually grounded but only moderately helpful to users. This may mean that while responses are well-structured and accurate, they do not always address the user's intent effectively. Optimizing intent recognition and tailoring responses to queries may help.
            
            ---
            Now interpret the following:
            Task: {self.parsed_request['task']}
            Metric Results: {metric_results}
            '''

            response = self.cfg.groq_client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response.choices[0].message.content.strip()

        return retry_with_backoff(call)