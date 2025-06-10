import ast
from evalbench.agents.interpretation import Interpretation
from evalbench.agents.module_selection import ModuleSelection
from evalbench.agents.recommendation import Recommendation
from evalbench.runtime_setup.runtime import get_config
import evalbench.utils.agent_helper as helper
from evalbench.utils.output_control import generate_report

class Master:
    def __init__(self):
        self.recommendation_agent = None
        self.interpretation_agent = None
        self.module_selector_agent = None
        self.cfg = get_config()
        self.request = {}

    def handle_user_request(self, instruction, data=None, eval_results=None, interpretation=None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be a non-empty string that instructs the agent to perform a tas.')

        steps_to_execute = ast.literal_eval(helper.plan_steps(instruction)) # identify the steps to execute (evaluation/interpretation/recommendation)
        task = helper.get_task(instruction, data) # to assist in interpretation and recommendation
        input_data = helper.parse_data(steps_to_execute, data) # parse data in the required form for downstream tasks

        self.request = {
            'instruction': instruction,
            'steps': steps_to_execute,
            'task': task,
            'data': input_data,
            'results': eval_results,
            'interpretation': interpretation,
        }

    def create_sub_agents(self):
        self.module_selector_agent = ModuleSelection(self.request)
        self.interpretation_agent = Interpretation(self.request)
        self.recommendation_agent = Recommendation(self.request)

    def execute(self):
        results = None
        interpretation = None
        recommendations = None

        steps_to_execute = self.request['steps']
        for step in steps_to_execute:
            if step == 'evaluation':
                results = self.module_selector_agent.execute()
                if not results:
                    raise ValueError(helper.improve_prompt(self.request['instruction']))
            elif step == 'interpretation':
                if not results and  not self.request['results']:
                    raise ValueError('Evaluation results required for interpretation. Please ensure that the evaluation step is executed before interpretation or provide results explicitly.')
                interpretation = self.interpretation_agent.interpret(results)
            elif step == 'recommendation':
                if not results and not self.request['results']:
                    raise ValueError('Evaluation results required for recommendation. Please ensure that the evaluation step is executed or provide results explicitly.')
                recommendations = self.recommendation_agent.recommend(results, interpretation)
            else:
                raise ValueError(helper.improve_prompt(self.request['instruction']))

        report_data = {
            'instruction': self.request['instruction'],
            'task': self.request['task'],
            'data': self.request['data'],
            'results': self.request['results'] if self.request['results'] else results,
            'interpretation': self.request['interpretation'] if self.request['interpretation'] else interpretation,
            'recommendations': recommendations
        }
        report = generate_report(report_data)

        return report