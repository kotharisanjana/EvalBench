import re
import json
from collections import defaultdict
from evalbench.agents.interpretation import Interpretation
from evalbench.agents.module_selection import ModuleSelection
from evalbench.agents.recommendation import Recommendation
from evalbench.runtime_setup.runtime import get_config

class Master:
    def __init__(self):
        self.recommendation_agent = None
        self.interpretation_agent = None
        self.module_selector_agent = None
        self.cfg = get_config()
        self.parsed_request = {}

    def handle_user_request(self, user_input):
        user_intent = self.classify_intent(user_input)
        user_input_data = self.extract_data(user_input)
        self.parsed_request = {
            'goal': user_intent,
            'data': user_input_data
        }

    def create_sub_agents(self):
        self.module_selector_agent = ModuleSelection(self.parsed_request)
        self.interpretation_agent = Interpretation(self.parsed_request)
        self.recommendation_agent = Recommendation()

    def execute(self):
        goal = self.parsed_request['goal']
        if goal == 'full_evaluation':
            eval_results = self.module_selector_agent.execute()
            interpretation = self.interpretation_agent.interpret(eval_results)
            recommendations = self.recommendation_agent.generate_recommendations(self.parsed_request, eval_results, interpretation)
            final_report = {
                'interpretation': interpretation,
                'recommendations': recommendations,
                'eval_results': eval_results
            }
            return final_report
        elif goal == 'interpretation_only':
            return self.interpretation_agent.interpret(self.parsed_request)
        elif goal == 'recommendation_only':
            return self.recommendation_agent.generate_recommendations(self.parsed_request)
        elif goal == 'interpretation and recommendation':
            interpretation = self.interpretation_agent.interpret(self.parsed_request)
            return self.recommendation_agent.generate_recommendations(self.parsed_request, interpretation)
        else:
            raise ValueError('Sorry, I couldn’t understand your request')

    # make confident decisions so downstream processes are deterministic and manageable
    def classify_intent(self, user_input):
        try:
            prompt = f'''
            You are an intent classification assistant. Your task is to understand what the user wants from this evaluation library.
    
            Given the user input below, classify the intent into **one and only one** of the following categories exactly as written:
    
            - full_evaluation
            - interpretation_only
            - recommendation_only
            - interpretation and recommendation
            - vague/unclear
    
            Do NOT add any explanation or extra text, just respond with exactly one of the above.
    
            User Input:
            \'\'\'{user_input}\'\'\'
            '''
            response = self.cfg.groq_client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
            )
            intent = response.choices[0].message['content'].strip()
        except Exception as e:
            intent = None

        return intent

    def extract_data(self, user_input):
        json_candidates = re.findall(r'(\{.*?}|\[.*?])', user_input, re.DOTALL)

        for blob in json_candidates:
            try:
                parsed = json.loads(blob)

                # batch mode
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    keys = set(parsed[0].keys())
                    if not all(set(item.keys()) == keys for item in parsed):
                        raise ValueError('Inconsistent keys across batch examples.')

                    input_data = defaultdict(list)
                    for item in parsed:
                        for key in keys:
                            input_data[key].append(item.get(key, ''))

                    return {
                        'input_data': dict(input_data),
                        'eval_results': None
                    }

                # single example mode/ evaluation mode
                elif isinstance(parsed, dict):
                    input_data_keys = {'query', 'response', 'context', 'reference', 'generated', 'retrieved_docs', 'relevant_docs'}
                    if all(k.lower() in input_data_keys for k in parsed.keys()):
                        return {
                            'input_data': parsed,
                            'eval_results': None
                        }
                    else:
                        input_data = {k: [v] for k, v in parsed.items()}
                        return {
                            'input_data': None,
                            'eval_results': input_data
                        }

                # there can also be a scenario where the user provides both input_data and eval_results

                else:
                    raise ValueError('Missing input data / Unable to extract valid input data. Please ensure your input is in the correct JSON format with required fields.')

            except json.JSONDecodeError:
                break

        return {
            'input_data': None,
            'eval_results': None
        }





