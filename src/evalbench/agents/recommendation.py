from jaraco.functools import retry

from evalbench.runtime_setup.runtime import get_config
from evalbench.utils.agent_helper import retry_with_backoff


class Recommendation:
    def __init__(self, parsed_request):
        self.cfg = get_config()
        self.parsed_request = parsed_request

    def recommend(self, results=None, interpretation=None):
        def call():
            prompt = f'''
            You are a language evaluation recommendation assistant.
            Your job is to analyze evaluation results, understand the interpretation if available and offer concrete, actionable suggestions to improve the model or system under evaluation.
            
            Guidelines:
            - Your goal is to suggest 2–3 specific and actionable recommendations to improve model performance. 
            - Tailor your suggestions based on the task type if provided. Use the interpretation (if present) to avoid restating analysis and focus only on what should be done next.
            - Respond with your recommendations in a concise numbered list.
            - Provide clear, practical steps that can be taken to address the issues identified in the results.
            - Respond ONLY with a numbered list of 2–3 actionable recommendations. Do NOT restate analysis or include explanations.
            
            Example 1:
            Task: summarization  
            Metric Results:
            {{
              'bleu_score': [0.22, 0.20, 0.19],
              'rouge_score': [0.31, 0.29, 0.30],
              'bert_score': [0.74, 0.76, 0.75]
            }}  
            Interpretation:
            The summaries are semantically relevant but diverge from reference phrasing. Lexical overlap is low.
            Recommendations:
            1. Fine-tune the summarization model on a domain-specific corpus to increase overlap with references.
            2. Introduce constraints or templates to guide phrasing in a more reference-aligned way.
            3. Evaluate with newer semantic metrics (e.g., QuestEval) to better optimize for meaning preservation.
            
            Example 2:
            Task: retrieval  
            Metric Results:
            {{
              'recall_at_k': [0.95, 0.96, 0.94],
              'precision_at_k': [0.41, 0.39, 0.40],
              'mrr_score': [0.34, 0.36, 0.33]
            }}  
            Interpretation:
            High recall but low precision and ranking quality. Many irrelevant documents retrieved, and relevant ones often ranked low.
            Recommendations:
            1. Improve document reranking with a cross-encoder or hybrid retrieval model.
            2. Apply post-retrieval filtering using answer relevance or passage entailment.
            3. Retrain embedding models with hard negatives to better separate relevant/irrelevant content.
            
            Example 3:
            Task: chatbot  
            Metric Results:
            {{
              'helpfulness_score': [0.54, 0.58, 0.55],
              'coherence_score': [0.84, 0.81, 0.83],
              'factuality_score': [0.64, 0.66, 0.62]
            }}  
            Interpretation:
            Responses are coherent and factual but only moderately helpful.
            Recommendations:
            1. Improve intent detection or user need classification to better tailor responses.
            2. Fine-tune on helpfulness-labeled datasets (e.g., Helpful/Harmless, OpenAssistant).
            3. Add retrieval augmentation for grounding responses in task-specific knowledge.
            
            Now make recommendations for the following:
            Task: {self.parsed_request['task'] if self.parsed_request['task'] else 'Unknown'}  
            Metric Results:  
            {results if results else self.parsed_request['results']}  
            Interpretation:  
            {interpretation if interpretation else self.parsed_request['interpretation']}
            '''

            response = self.cfg.groq_client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=1,
            )
            return response.choices[0].message.content.strip()

        return retry_with_backoff(call)



