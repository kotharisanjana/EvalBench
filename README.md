# EvalBench

#### End-to-End Flow for EvalBench:
1. Prepare Test Suite: Create a test_suite.jsonl file containing the test cases (input questions, expected answers, and optionally gold sources for verification).
2. Prepare Predictions: Create a predictions.jsonl file with model-generated answers corresponding to the test cases.
3. Install EvalBench: Set up the EvalBench CLI tool on your system, ensuring all dependencies are installed.
4. Run Evaluation: Execute the evalbench run command in the terminal, passing the paths to your test_suite.jsonl and predictions.jsonl, specifying model, evaluation metrics, and critique option.
5. Model Prediction Evaluation: The tool processes the input and predictions, computing metrics such as accuracy, relevance, and hallucination.
6. Critique Agent Activation: If hallucination detection is enabled, the CritiqueAgent (e.g., GPT-4) is called to analyze incorrect predictions, identify root causes (e.g., vague, unsupported claims), and suggest actionable fixes.
7. Generate Structured Report: After evaluating each test case, the tool outputs a structured evaluation report.

#### Core Use Cases This Tool Can Support
1. LLM QA Evaluation (RAG or fine-tuned models)
2. Semantic Search Evaluation (based on query + result ranking)
3. Chatbot Response Quality (user query → response + context)
4. Knowledge Testing / MCQ Accuracy
5. Open-ended Q&A (e.g., interview practice, medical advice, etc.)
