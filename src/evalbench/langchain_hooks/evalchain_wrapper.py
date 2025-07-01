from langchain_core.runnables import Runnable
from evalbench.runtime_setup.runtime import set_config
from evalbench.metrics.evaluate import evaluate_module, evaluate_metric

class EvalChain(Runnable):
    def __init__(self, runnable, metrics=None, module=None, evalbench_config=None, **kwargs):
        self.runnable = runnable
        self.metrics = metrics
        self.module = module
        self.input_params = kwargs
        if evalbench_config:
            set_config(evalbench_config)

    def invoke(self, input, config=None, **kwargs):
        output = self.runnable.invoke(input, config, **kwargs)

        input_params = self.input_params.copy()
        if isinstance(output, dict):
            llm_response = output.get('text') or output.get('response') or output
        else:
            llm_response = output
        input_params['llm_response'] = [llm_response] if isinstance(llm_response, str) else llm_response

        if self.module:
            results = evaluate_module(self.module, **input_params)
            return results
        elif self.metrics:
            results = [
                evaluate_metric(metric, **input_params)
                for metric in self.metrics
                ]
            return results
        else:
            results = None

        return {
            "output": output,
            "eval": results,
        }

    # override batch()

