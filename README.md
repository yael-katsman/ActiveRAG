# Bridging Retrieval and Reasoning: Enhancing the Associate Agent in ACTIVERAG for Knowledge-Intensive Tasks

# Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Guide](#setup-guide)  
    3.1 [Installation](#installation)  
    3.2 [Reproduction](#reproduction)
4. [File Description](#file-description)  
    4.1 [Script Files](#script-files)  
    4.2 [Build Files](#build-files)  
    4.3 [Evaluate Files](#evaluate-files)  
    4.4 [Result Folders](#result-folders)  
    4.5 [Prompts](#prompts)
5. [Baseline Models](#baseline-models)
6. [Results](#results)  
    6.1 [Accuracy Comparison](#accuracy-comparison)  
    6.2 [BLEU Score Comparison](#bleu-score-comparison)
7. [Contact](#contact)


## Overview

ACTIVERAG is an innovative framework designed to enhance LLMs by shifting from passive knowledge acquisition to an active learning approach. Unlike conventional RAG models, ACTIVERAG dynamically integrates external knowledge with the model’s existing understanding through a multi-agent system. This system includes the Anchoring agent, which grounds information in relevant contexts; the Logician, which maintains logical coherence; the Cognition agent, which aligns new data with the model’s existing knowledge structures; and the Associate agent, which links new information with prior knowledge to enhance reasoning. 

Our project focuses on enhancing the Associate agent within ACTIVERAG by employing prompt engineering techniques such as iterative refinement, chain of thought prompting, and role assignment. We developed two modified prompts and tested these enhancements on the Natural Questions (NQ) and TriviaQA datasets.

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



## Prerequisites
To use ActiveRAG, you will need the following dependencies installed on your machine:

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
### **Script Files**
The scripts are used for running and evaluating the ActiveRAG model, VanillaRAG, Chain of Thought (COT), and the LLM's GPT 4.0 Turbo and GPT 4.0-mini models.

#### **Run Files (Model Execution)**
These files are used to run the models:
- `run.py`: Runs the **ActiveRAG** model.
- `Vanila_run.py`: Runs the **VanillaRAG** model.
- `COT_run.py`: Runs the **COT** model.
- `API_run.py`: Runs **LLM models** according to the selected model in the code.

#### **Build Files (Result Preparation)**
These files build CSV files with model results, which are then used to calculate Accuracy or BLEU scores.

**For Accuracy:**
- `build.py`: General build script for accuracy.
- `API_build.py`: Build script for **LLM** model.
- `COT_build.py`: Build script for **COT** model.
- `vanila_build.py`: Build script for **VanillaRAG** model.

**For BLEU:**
- `build_bleu.py`: General build script for BLEU score.
- `API_build_bleu.py`: Build script for **LLM** model.
- `COT_build_bleu.py`: Build script for **COT** model.
- `vanila_build_bleu.py`: Build script for **VanillaRAG** model.

#### **Evaluate Files (Result Calculation)**
These files are used to calculate the accuracy or BLEU scores.

**For Accuracy:**
- `evaluate.py`: General evaluation script for accuracy.
- `API_evaluate.py`: Accuracy evaluation for **LLM** model.
- `COT_evaluate.py`: Accuracy evaluation for **COT** model.
- `vanila_evaluate.py`: Accuracy evaluation for **VanillaRAG** model.

**For BLEU:**
- `evaluate_bleu.py`: General evaluation script for BLEU score.
- `API_evaluate_bleu.py`: BLEU evaluation for **LLM** model.
- `COT_evaluate_bleu.py`: BLEU evaluation for **COT** model.
- `vanila_evaluate_bleu.py`: BLEU evaluation for **VanillaRAG** model.
- 
### **Result Folders**
These folders contain the log files generated from running the models.

- **Log**: Files created from running **ActiveRAG** with the original prompt.
- **log_1**: Files created from running **ActiveRAG** with prompt 1.
- **log_2**: Files created from running **ActiveRAG** with prompt 2.
- **vanila**: Files created from running **VanillaRAG**.
- **cot**: Files created from running **COT**.
- **api_4**: Files created from running the **GPT 4.0-turbo** model.
- **api_4_mini**: Files created from running the **GPT 4.0-mini** model.

### **Prompts**
The prompts used for *ActiveRAG* can be found in the *SRC* folder.

## Baseline Models
Our project used several baseline models to evaluate performance:
1. Vanilla RAG: A standard RAG model that retrieves relevant passages and generates responses.
2.	Chain-of-Thought (CoT): A reasoning approach that prompts models to generate step-by-step explanations.
3.	GPT-4.0 Mini Model: A budget-friendly, multimodal variant of GPT-4 designed for smaller tasks, focusing on cost-efficiency and speed, primarily used in our experiments.
4.	GPT-4 Turbo: A high-performance variant of GPT-4 that offers superior accuracy and output quality, used for additional evaluations in scenarios where accuracy is critical.


## Results
Our experiments focused on evaluating ACTIVERAG’s performance in Top-5 and Top-10 settings, where the model generates responses using the top 5 or top 10 passages retrieved by an external retrieval model. 
               |

Table 4 compares the best BLEU scores of baseline models with those of ACTIVERAG (with the Associate agent), with the best performance achieved using Prompt 1 for NQ and Prompt 2 for TriviaQA. The table shows that the ACTIVERAG achieved BLEU scores of 0.32 on NQ and 0.36 on TriviaQA, outperforming all other models on NQ, including GPT-4 Turbo (0.187). Although Vanilla RAG achieved a higher BLEU score on TriviaQA (0.46), ACTIVERAG's performance was still competitive and significantly better than CoT and GPT-4.0 Mini. These results highlight the effectiveness of the targeted prompt modifications, demonstrating superior response quality and alignment with reference answers, especially on NQ.

## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***





results: (טבלאות זמניות לראות את הנתונים)





