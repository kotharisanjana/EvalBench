from evalbench.agents.interpretation import Interpretation
from evalbench.agents.module_selection import ModuleSelection
from evalbench.agents.recommendation import Recommendation
from evalbench.runtime_setup.runtime import get_config
import evalbench.utils.agent_helper as helper
from evalbench.utils.metrics_helper import generate_report

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

        intent = helper.get_user_intent(instruction) # identify the steps to execute (evaluation/interpretation/recommendation)
        task = helper.get_task(instruction, data) # to assist in interpretation and recommendation
        input_data = helper.parse_data(intent, data) # parse data in the required form for downstream tasks

        self.request = {
            'instruction': instruction,
            'intent': intent,
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

        intent = self.request['intent']
        if intent == 'full_pipeline':
            results = self.module_selector_agent.execute()
            if results:
                interpretation = self.interpretation_agent.interpret(results)
                recommendations = self.recommendation_agent.recommend(results, interpretation)
            # if metrics not found, results would be none - suggest improving the prompt
            else:
                raise ValueError(helper.improve_prompt(self.request['instruction']))
        elif intent == 'evaluation_only':
            results = self.module_selector_agent.execute()
            if not results:
                raise ValueError(helper.improve_prompt(self.request['instruction']))
        elif intent == 'interpretation_only':
            if not self.request['results']:
                raise ValueError('Results must be provided for interpretation.')
            interpretation = self.interpretation_agent.interpret()
        elif intent == 'recommendation_only':
            if not self.request['results']:
                raise ValueError('Results must be provided for recommendations.')
            recommendations = self.recommendation_agent.recommend()
        elif intent == 'interpretation and recommendation':
            if not self.request['results']:
                raise ValueError('Results must be provided for interpretation and recommendation.')
            interpretation = self.interpretation_agent.interpret()
            recommendations = self.recommendation_agent.recommend(interpretation)
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