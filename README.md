# AURA - ActiveRAG Unified Agent Model

# Table of Contents




## Overview

ACTIVERAG is an innovative framework designed to enhance LLMs by shifting from passive knowledge acquisition to an active learning approach. Unlike conventional RAG models, ACTIVERAG dynamically integrates external knowledge with the model’s existing understanding through a multi-agent system. This system includes the Anchoring agent, which grounds information in relevant contexts; the Logician, which maintains logical coherence; the Cognition agent, which aligns new data with the model’s existing knowledge structures; and the Associate agent, which links new information with prior knowledge to enhance reasoning. 

Our project focuses on enhancing the Associate agent within ACTIVERAG by employing prompt engineering techniques such as iterative refinement, chain of thought prompting, and role assignment. We developed two modified prompts and tested these enhancements on the Natural Questions (NQ) and TriviaQA datasets.



## Prerequisites
You will need the following dependencies installed on your machine:

Python 3.8 or higher

## Setup Guide
### Installation
To reproduce the results of this project, you can clone the repository and install the required dependencies:

```bash
git clone https://github.com/hillysegal1/ActiveRAG
pip install -r requirements.txt
```
### Reproduction
You can reproduce the results from our paper using the following command:
```bash
python -m logs.eval --dataset nq --topk 5
```
We also provide the full request code, you can re-request for further exploration.

First you need to enter your openai api key:

```python
API_KEY="your-key"
```

Then, run the following script:

```bash
python -m scripts.run --dataset nq --topk 5
```
Analyzing log files:

```bash
python -m scripts.build --dataset nq --topk 5
```

Evaluate:

```bash
python -m scripts.evaluate --dataset nq --topk 5
```
To change the model or the evaluation metrics, run the scripts according to the file description below. 

## **File description**


### **Prompts**
The prompts used for *ActiveRAG* can be found in the *SRC* folder.

## Baseline Models
Our project used several baseline models to evaluate performance:
1. Vanilla RAG: A standard RAG model that retrieves relevant passages and generates responses.
2.	Chain-of-Thought (CoT): A reasoning approach that prompts models to generate step-by-step explanations.
3.	GPT-4.0 Mini Model: A budget-friendly, multimodal variant of GPT-4 designed for smaller tasks, focusing on cost-efficiency and speed, primarily used in our experiments.
4.	GPT-4 Turbo: A high-performance variant of GPT-4 that offers superior accuracy and output quality, used for additional evaluations in scenarios where accuracy is critical.


## Results
## models
### Small models

| Dataset   | Model     | Retrieval Setting | Epochs | Learning Rate | Loss Function    |
|-----------|-----------|-------------------|--------|---------------|------------------|
| NQ        | RoBERTa   | Top 5            | 10     | 0.001         | MSELoss          |
| PopQA     | RoBERTa   | Top 5            | 10     | 0.01          | CrossEntropyLoss |
| TriviaQA  | RoBERTa   | Top 5            | 10     | 0.001         | MSELoss          |
| WebQA     | RoBERTa   | Top 5            | 10     | 0.01          | CrossEntropyLoss |
| NQ        | RoBERTa   | Top 10           | 10     | 0.005         | MSELoss          |
| PopQA     | BERT      | Top 10           | 10     | 0.01          | CrossEntropyLoss |
| TriviaQA  | RoBERTa   | Top 10           | 10     | 0.01          | CrossEntropyLoss |
| WebQA     | BERT      | Top 10           | 10     | 0.01          | CrossEntropyLoss |

### Big models

| Retrieval Setting | Learning Rate | Epochs | Loss Function | Optimizer |
|-------------------|---------------|--------|---------------|-----------|
| Top 10           | 0.01          | 10     | MSELoss       | AdamW     |
| Top 5            | 0.001         | 10     | MSELoss       | AdamW     |

Table 1-Accuracy Comparison of ACTIVERAG Agents and AURA Models

## Contact
For any questions or issues regarding this project, feel free to contact us at  [yael-k@campus.technion.ac.il],[hillysegal@campus.technion.ac.il] ,[ronshahar@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547v1)***








