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
        intent = self.request['intent']
        if intent == 'full_evaluation':
            results = self.module_selector_agent.execute()
            interpretation = self.interpretation_agent.interpret(results)
            recommendations = self.recommendation_agent.recommend(results, interpretation)
            final_report = {
                'results': results,
                'interpretation': interpretation,
                'recommendations': recommendations,
            }
            return final_report
        elif intent == 'interpretation_only':
            return self.interpretation_agent.interpret()
        elif intent == 'recommendation_only':
            return self.recommendation_agent.recommend()
        elif intent == 'interpretation and recommendation':
            interpretation = self.interpretation_agent.interpret()
            recommendations = self.recommendation_agent.recommend(interpretation)
            final_report = {
                'interpretation': interpretation,
                'recommendations': recommendations,
            }
            return final_report
        else:
            raise ValueError('Sorry, I couldn’t understand your request')