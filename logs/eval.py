import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--topk', type=int, required=True)
args = parser.parse_args()

dataset = args.dataset
topk = args.topk

num_anchoring = 0
num_associate=0
num_logician=0
num_cognition=0


for i in range(500):
        file_path = f'logs/{dataset}/top{topk}/{i}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
            if data['anchoring']['anchoring_correctness'] == 'True':
                num_anchoring+=1
            if data['associate']['associate_correctness'] == 'True':
                num_associate+=1
            if data['logician']['logician_correctness'] == 'True':
                num_logician+=1
            if data['cognition']['cognition_correctness'] == 'True':
                num_cognition+=1

print(f"anchoring: {num_anchoring/500}")
print(f"associate: {num_associate/500}")
print(f"logician: {num_logician/500}")
print(f"cognition: {num_cognition/500}")