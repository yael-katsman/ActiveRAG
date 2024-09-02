# vanilla_plan.py

from vanilla_agent import VanillaRAGAgent

class VanillaRAGPlan:
    def __init__(self, agent: VanillaRAGAgent):
        self.agent = agent

    def create_vanilla_rag_plan(self, question: str, retrieved_passages: str):
        """
        Create a plan to execute VanillaRAG using the given agent, question, and passages.
        """
        self.agent.send_vanilla_rag_message(retrieved_passages, question)

    def execute(self):
        """
        Execute the plan.
        """
        print("Executing VanillaRAG plan...")
        self.agent.send_vanilla_rag_message()

    def get_result(self):
        """
        Get the result from the agent after executing the plan.
        """
        return self.agent.get_output()
