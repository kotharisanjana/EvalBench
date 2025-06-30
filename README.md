# EvalBench 
`EvalBench` is a plug-and-play Python package for evaluating outputs of large language models (LLMs) across a variety of metrics - from response quality and retrieval accuracy to hallucination and prompt alignment.

It now includes agentic workflows: just describe what you want to understand or improve about your LLM outputs, and EvalBench will plan and execute a tailored sequence of evaluation, interpretation, and recommendation steps — automatically!

### 🚀 Key Features:
- 18+ built-in metrics covering coherence, relevance, hallucination, BLEU, ROUGE, MRR, and more
- User-defined custom metrics with a simple decorator-based API
- Modular architecture to group related metrics and share inputs
- Agentic execution: EvalBench can reason about your goal and execute the necessary steps (evaluate → interpret → recommend)
- Batch support, configurable output (print/save), and JSON-compatible results

### 📊 Modules and Metric Categories:

| Module               | Metrics                                                                      | 
|----------------------|------------------------------------------------------------------------------|
| response_quality     | conciseness_score, coherence_score, factuality_score                         | 
| reference_based      | bleu_score, rouge_score, meteor_score, semantic_similarity_score, bert_score | 
| contextual_generation | faithfulness_score, hallucination_score, groundedness_score                  | 
| retrieval          | recall_at_k_score, precision_at_k_score, ndcg_at_k_score, mrr_score          | 
| query_alignment       | context_relevance_score                                                      | 
| response_alignment    | response_relevance_score, response_helpfulness_score                         | 
| user defined module               | User-registered custom metrics                                               | 

### 🧠 Agentic Workflow:
EvalBench follows a three-step agentic pipeline, automatically triggered based on user instructions:

1. Evaluation – Runs relevant metrics to score model outputs. EvalBench intelligently selects which metrics to use if not explicitly specified.
2. Interpretation – Analyzes the evaluation results and highlights potential issues with model behavior.
3. Recommendation – Suggests improvements to prompts, model setup, data inputs, or evaluation strategy.

Just write your request in plain language — EvalBench will take care of the rest.

---

## 🚀 Usage
```bash
pip install evalbench
```

All usage examples, including how to write your own custom metrics and how to use the agentic pipeline in practice, are available in this Jupyter notebook:

👉 [View the Notebook](https://colab.research.google.com/drive/1Y0oSzgPahpANlTbfrbTz9aMPNbr_3H2e#scrollTo=8gUQe2G7VDQ1)

---

## 💡 Use Cases
EvalBench is ideal for:
- Evaluating LLM apps like summarizers, chatbots, and search agents using built-in metrics
- Integrating custom, domain-specific metrics into the EvalBench's ecosystem
- Getting automatic eval → interpret → recommend pipelines from natural language instructions
- Rapidly iterating on model outputs, prompts, and evaluation strategies
  
---

## 🚧 Coming Soon
- Ecosystem integration - langchain hook
- CLI support
