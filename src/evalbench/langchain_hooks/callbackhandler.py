from langchain.callbacks import BaseCallbackHandler
from evalbench.runtime_setup.runtime import set_config
from evalbench.metrics.evaluate import evaluate_module, evaluate_metric
from evalbench.error_handling.custom_error import Error, ErrorMessages

class CallbackHandler(BaseCallbackHandler):
    def __init__(self, metrics=None, module=None, evalbench_config=None, **kwargs):
        self.metrics = metrics
        self.module = module
        self.input_params = kwargs
        if evalbench_config:
            set_config(evalbench_config)

    # on_llm_end() is called every time an LLM generates an output — evaluate the output using evalbench
    def on_llm_end(self, response, **kwargs):
        llm_response = [gen.text for generation in response.generations for gen in generation]

        input_params = self.input_params.copy()
        input_params['llm_response'] = [llm_response] if isinstance(llm_response, str) else llm_response

        if self.module:
            return evaluate_module(self.module, **input_params)
        elif self.metrics:
            return [
                evaluate_metric(metric, **input_params)
                for metric in self.metrics
            ]
        else:
            raise Error(ErrorMessages.CUSTOM_ERROR , message='No evaluation metrics/module specified in the callback handler.')