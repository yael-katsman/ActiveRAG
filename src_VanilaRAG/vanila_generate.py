# vanilla_generate.py

from vanilla_agent import VanillaRAGAgent
from vanilla_plan import VanillaRAGPlan

def run_vanilla_rag_experiment(question: str, retrieved_passages: str):
    """
    Run the VanillaRAG experiment with the provided question and retrieved passages.
    """
    # Initialize the VanillaRAG agent
    vanilla_rag_agent = VanillaRAGAgent()

    # Set up the VanillaRAG plan
    vanilla_rag_plan = VanillaRAGPlan(vanilla_rag_agent)
    vanilla_rag_plan.create_vanilla_rag_plan(question, retrieved_passages)

    # Execute the plan
    vanilla_rag_plan.execute()

    # Get and print the result
    result = vanilla_rag_plan.get_result()
    print(f"VanillaRAG Result: {result}")
    return result


