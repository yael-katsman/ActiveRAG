import sys
import os
import json
import time
from argparse import ArgumentParser
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import traceback


# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('API_KEY')


class VanillaAgent:
    def __init__(self, model='gpt-4-turbo'):
        self.message = []
        self.model = model

    def ask_question(self, question):
        """
        Send the question directly to OpenAI API without using retrieved passages.
        """
        message = {
            'role': 'user',
            'content': f"Question: {question}"
        }
        self.message.append(message)

        try:
            ans = openai.ChatCompletion.create(
                model=self.model,
                messages=self.message,
                temperature=0.2,
                n=1
            )
            self.parse_message(ans)
            return ans
        except Exception as e:
            print(e)
            time.sleep(20)
            ans = openai.ChatCompletion.create(
                model=self.model,
                messages=self.message,
                temperature=0.2,
                n=1
            )
            self.parse_message(ans)
            return ans

    def parse_message(self, completion):
        content = completion['choices'][0]['message']['content']
        role = completion['choices'][0]['message']['role']
        record = {'role': role, 'content': content}
        self.message.append(record)
        return record

    def get_output(self):
        """
        Get the latest output generated by the model.
        """
        if len(self.message) == 0:
            raise ValueError("No messages found. Cannot get output.")
        return self.message[-1]['content']


def run_api_experiment(question: str):
    """
    Ask the API the question directly and get the result.
    """
    # Initialize the VanillaAgent (without RAG or retrieval)
    vanilla_agent = VanillaAgent()

    # Ask the question to the API
    vanilla_agent.ask_question(question)

    # Get and print the result
    result = vanilla_agent.get_output()
    print(f"API Result: {result}")
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    dataset = args.dataset
    filename = f'data/data_{dataset}_sampled.jsonl'

    api_directory = f'api_4/{dataset}/results'

    # Create directory for logs if it doesn't exist
    if not os.path.exists(api_directory):
        os.makedirs(api_directory)

    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file)):
            try:
                data = json.loads(line)
                question = data['question']

                # Ask the API the question directly
                api_result = run_api_experiment(question)

                # Save the result
                with open(f'{api_directory}/{dataset}_idx_{i}.json', 'w', encoding='utf-8') as result_file:
                    json.dump({'question': question, 'api_result': api_result}, result_file)

            except Exception as e:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Error at index {i} - {current_time}: {e}")
                traceback.print_exc()
                break