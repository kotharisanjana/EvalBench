from evalbench.agents.interpretation import Interpretation
from evalbench.agents.module_selection import ModuleSelection
from evalbench.agents.recommendation import Recommendation
from evalbench.runtime_setup.runtime import get_config
import evalbench.utils.agent_helper as helper

class Master:
    def __init__(self):
        self.recommendation_agent = None
        self.interpretation_agent = None
        self.module_selector_agent = None
        self.cfg = get_config()
        self.request = {}

    def handle_user_request(self, instruction, data=None, eval_results=None, interpretation=None):
        intent = helper.get_user_intent(instruction) # identify the steps to execute (evaluation/interpretation/recommendation)
        task = helper.get_task(instruction, data) # to assist in interpretation and recommendation
        input_data = helper.parse_data(data) # parse data in the required form for downstream tasks
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
        if intent == 'full_evaluation':
            results = self.module_selector_agent.execute()
            interpretation = self.interpretation_agent.interpret(results)
            recommendations = self.recommendation_agent.recommend(results, interpretation)
        elif intent == 'interpretation_only':
            interpretation = self.interpretation_agent.interpret()
        elif intent == 'recommendation_only':
            recommendations = self.recommendation_agent.recommend()
        elif intent == 'interpretation and recommendation':
            interpretation = self.interpretation_agent.interpret()
            recommendations = self.recommendation_agent.recommend(interpretation)
        else:
            raise ValueError('Sorry, I couldn’t understand your request')

        report_data = {
            'instruction': self.request['instruction'],
            'task': self.request['task'],
            'results': self.request['results'] if self.request['results'] else results,
            'interpretation': self.request['interpretation'] if self.request['interpretation'] else interpretation,
            'recommendations': recommendations
        }
        report = helper.generate_report(report_data)

        return report