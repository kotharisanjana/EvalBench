# EvalBench Usage Guide

EvalBench is a flexible evaluation framework for LLM applications, offering predefined and custom metrics, easy configuration, and multiple evaluation modes.

---

### Initialize Configuration

```python
import evalbench as eb

# Create and apply evaluation configuration
config = eb.EvalConfig(groq_api_key="", output_mode="print")
eb.set_config(config)
```

### Usage Examples
#### 1. Evaluate a single predefined metric

```python
response = ["A binary search algorithm reduces the time complexity to O(log n)."]
context = [["Binary search works on sorted arrays and is faster than linear search."]]

eb.faithfulness_score(context=context, generated=response)
```

#### 2. Evaluate all metrics in a predefined module

```python
eb.evaluate_module(
    module=["contextual_generation"],
    context=context,
    generated=response,
)
```

#### 3. Register custom metrics
Create a custom metric in a file (eg. custom.py):

```python
from evalbench import register_metric, handle_output

@register_metric(name="len_metric", module="custom", required_args=["response", "reference"])
@handle_output()
def my_custom_metric(response, reference):
    return [len(response) / len(reference)]
```

#### 4. Load custom metric file and evaluate

```python
eb.load_custom_metrics("custom.py")

response = ["A binary search algorithm reduces the time complexity to O(log n).", "The Eiffel Tower is located in Berlin and was built in the 1800s."]
reference = ["Binary search works on sorted arrays and is faster than linear search.", "In Python, a generator yields items one at a time using the 'yield' keyword."]

# Evaluate custom metric directly
eb.my_custom_metric(response, reference)

# Evaluate all metrics in the 'custom' module
eb.evaluate_module(
    module=["custom"],
    response=response,
    reference=reference,
)
```

