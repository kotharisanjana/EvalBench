from evalbench.runtime_setup.runtime import get_config

class Interpretation:
    def __init__(self, parsed_request):
        self.cfg = get_config()
        self.parsed_request = parsed_request

    def interpret(self, results=None):
        eval_results = self.parsed_request['data']['eval_results']

        metric_results = eval_results if eval_results is not None else results

        prompt = f'''
        You are an expert model evaluation analyst.

        Your task is to interpret a set of evaluation metric results computed over model outputs. Based on these scores, provide **concise, accurate, and insightful analysis** of the model’s **quality, strengths, and weaknesses**. Your goal is to help a practitioner understand what the numbers reveal about system performance, even without seeing the raw inputs.
        
        Instructions:
        - Focus on the **meaning and implications of each metric**.
        - Explain **patterns, tradeoffs, or inconsistencies** across metrics.
        - Avoid generic commentary; use **concrete, data-driven reasoning**.
        - If results suggest a strength (e.g. high coherence), say what that means functionally.
        - If results suggest a weakness (e.g. low factuality or MRR), suggest **why that may matter** in practice.
        - Keep the tone analytical and actionable. Do **not** just restate the scores.
        
        Here are some examples:
        Example 1:
        Metric Results:
        {{
          "bleu_score": [0.21, 0.25, 0.22],
          "rouge_score": [0.30, 0.33, 0.31],
          "bert_score": [0.78, 0.80, 0.77]
        }}
        Interpretation:
        The model produces moderately relevant and semantically close responses (high BERT score), but its lexical overlap is relatively low (lower BLEU/ROUGE), suggesting that while the meaning is preserved, phrasing diverges from references. It may be suitable for creative paraphrasing, but less so for strict template-matching tasks.
        
        ---
        
        Example 2:
        Metric Results:
        {{
          "recall_at_k": [0.92, 0.95, 0.90],
          "precision_at_k": [0.45, 0.50, 0.48],
          "mrr_score": [0.35, 0.40, 0.38]
        }}
        Interpretation:
        The system retrieves most of the relevant documents (high recall), but many of the retrieved results are irrelevant (low precision). Additionally, relevant documents often appear lower in the list (low MRR), indicating room for improvement in ranking. Consider improving ranking heuristics or embedding quality.
        
        ---
        
        Example 3:
        Metric Results:
        {{
          "helpfulness_score": [0.55, 0.48, 0.52],
          "coherence_score": [0.82, 0.85, 0.80],
          "factuality_score": [0.65, 0.62, 0.60]
        }}
        Interpretation:
        The chatbot's responses are mostly coherent and factually grounded but only moderately helpful to users. This may mean that while responses are well-structured and accurate, they do not always address the user's intent effectively. Optimizing intent recognition and tailoring responses to queries may help.
        
        ---

        Now interpret the following:
        
        Metric Results:
        {metric_results}
        '''

        try:
            response = self.cfg.groq_client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{'role': 'user', 'content': prompt}],
            )
            interpretation = response.choices[0].message['content'].strip()
        except Exception as e:
            interpretation = None

        return interpretation



