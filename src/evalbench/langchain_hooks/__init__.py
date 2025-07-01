# CallbackHandler
# How it works - Listens for LLM outputs via on_llm_end, then runs evalbench evaluation
# Where it's used - Attached to any LLM or Chain via callbacks
# Use case - evaluate LLM outputs in real-time during generation

# Tool (Agent Callable)
# How it works - Wraps evaluate_module() and evaluate_metric() as a LangChain @tool() callable by agents
# Where it's used - In agent tool lists
# Use case - Allows agents to use evaluation as tools in their workflows

# EvalChain (Wrapper)
# How it works - Wraps any Runnable, runs evaluation on its output, returns {output, eval}
# Where it's used - Around chains, prompts, or agents
# Use case - Adds evaluation as a middleware layer; returns structured outputs


