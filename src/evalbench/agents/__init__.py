# User → Master Agent → Module Selector Agent -> Evaluation Module
#         ↓                                             ↓
#   Interpretation Agent                         Evaluation Results
#         ↓
#   Recommendation Agent
#         ↓
#      Final Report

# 1. Master Agent (Supervisor / Orchestrator)
# Accepts natural language queries.
# Maintains conversational state/ autonomous clarification/ iterative reasoning or planning/ memory (future enhancements)
# Delegates tasks to sub-agents.

# 2. Module Selector Agent
# Interprets user goals.
# Selects Appropriate metrics (EvalBench modules/tools to call)
# Executes the module and returns results.

# 3. Interpretation Agent
# Takes in metric results
# Produces natural language explanation of what the results mean.
# Flags contradictions, metric tradeoffs, or uncertainties.

# 4. Recommendation Agent
# Takes original data + eval results + interpretation.
# Suggests how to improve performance.
