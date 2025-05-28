# EvalBench Usage Guide

EvalBench is a flexible evaluation framework for LLM applications, offering predefined and custom metrics, easy configuration, and multiple evaluation modes.

---

## Setup

### Initialize Configuration

```python
import evalbench as eb

# Create and apply evaluation configuration
config = eb.EvalConfig(groq_api_key='', output_mode='print')
eb.set_config(config)


Usage Examples
1. Evaluate a single predefined metric directly

import evalbench as eb

response = [["It is raining in Atlanta."]]
context = ["It is hot in Atlanta"]

eb.faithfulness_score(context=context, generated=response)

2. Evaluate all metrics in a predefined module

eb.evaluate_module(
    module=['contextual_generation'],
    context=context,
    generated=response,
)

3. Register a Custom Metric
Create a custom metric in a file custom.py:

from evalbench import register_metric, handle_output

@register_metric(name="len_metric", module="custom", required_args=['response', 'reference'])
@handle_output()
def my_custom_metric(response, reference):
    return [len(response) / len(reference)]

4. Load custom metric file and evaluate

e.load_custom_metrics('custom.py')

response = ["It is a rainy and stormy day in Atlanta"]
reference = [["It is hot in Atlanta"], ["It is amazing!"]]

# Evaluate custom metric directly
e.my_custom_metric(response, reference)

# Evaluate all metrics in the 'custom' module
results = e.evaluate_module(
    module=['custom'],
    response=response,
    reference=reference,
)
print(results)


