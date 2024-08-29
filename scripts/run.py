from src.generate import create_agent_group
from src.generate import create_plan
from argparse import ArgumentParser
from src.prompt import Prompt
from datetime import datetime
from tqdm import tqdm
import traceback
import time
import json
import os

parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset
    filename = f'data/data_{dataset}_sampled.jsonl'
    topk = args.topk

    directory = f'log_2/{dataset}/top{topk}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file)):
            try:
                data = json.loads(line)
                id = data['id']
                question = data['question']
                answers = data['answers']
                passages = data['passages'][:topk]


                input = {
                    'question':question,
                    'passages':passages,
                    '__answers__':answers,
                }

                print(f"{i}. {question}")
                time.sleep(5)
                plan = create_plan(create_agent_group(Prompt()),init_input=input)
                plan.excute()

                plan.save_log(f'log_2/{dataset}/top{topk}/{dataset}_idx_{i}.json')

            except Exception as e:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Error at index {i} - {current_time}: {e}")
                traceback.print_exc()
                break
