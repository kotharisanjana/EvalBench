from evalbench.agents.master import Master
import evalbench.utils.output_control as print_control

def run_agent_pipeline(instruction, data=None, results=None, interpretation=None):
    print_control.suppress_printing()

    master_agent = Master()
    master_agent.handle_user_request(instruction, data, results, interpretation)
    master_agent.create_sub_agents()
    report = master_agent.execute()
    return report
