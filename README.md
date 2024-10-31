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

**Table 1**: 
| Prompt       | NQ Top-5 Accuracy (%) | NQ Top-10 Accuracy (%) | TriviaQA Top-5 Accuracy (%) | TriviaQA Top-10 Accuracy (%) |
|--------------------|-----------------------|------------------------|-----------------------------|-------------------------------|
| Original Prompt    | 66.8                  | 68                     | 95                          | 95.4                         |
| Modified Prompt 1  | 64.6                  | 69.8                   | 94.8                        | 96                           |
| Modified Prompt 2  | 65.4                  | 68                     | 95.2                        | 95.6                         |

Table 1 shows the accuracy of the ACTIVERAG with the Associated agent, using the original and modified prompts. Modified Prompt 1, which emphasized deeper contextual understanding, improved accuracy in the Top-10 setting on both NQ and TriviaQA, suggesting better integration of contextual knowledge. However, it slightly reduced accuracy in the Top-5 setting. On the other hand, Modified Prompt 2, with a collaborative expert-based approach, showed moderate improvements on TriviaQA but no gains on NQ.

**Table 2**:
| Prompt       | NQ Top-5 BLEU | NQ Top-10 BLEU | TriviaQA Top-5 BLEU | TriviaQA Top-10 BLEU |
|--------------------|----------------|----------------|---------------------|-----------------------|
| Original Prompt    | 0.28           | 0.28           | 0.32                | 0.32                  |
| Modified Prompt 1  | 0.31           | 0.32           | 0.34                | 0.34                  |
| Modified Prompt 2  | 0.31           | 0.31           | 0.36                | 0.35                  |

Table 2 compares ACTIVERAG's (with the Associate agent) BLEU scores for the original and modified prompts. As shown in the table, Modified Prompt 1 consistently improved BLEU scores across both NQ and TriviaQA, demonstrating better contextual integration and response fluency. Modified Prompt 2 achieved the highest BLEU scores on TriviaQA, particularly in the Top-5 setting (0.36), but did not maintain the same level of improvement on NQ, highlighting that its benefits were more dataset-specific. This indicates that while both prompts enhanced language quality, their effectiveness varied depending on the context.

**Table 3**:
| Model            | Best Accuracy (NQ) (%) | Best Accuracy (TriviaQA) (%) |
|------------------|------------------------|------------------------------|
| Vanilla RAG      | 48.2                   | 85.6                         |
| CoT              | 55.2                   | 90.6                         |
| GPT-4.0 Mini     | 55.4                   | 88.0                         |
| GPT-4 Turbo      | 65.8                   | 93.8                         |
| ACTIVERAG (w. Associate Agent)       | 69.8                   | 96.0                         |

Table 3 compares the best accuracy results of baseline models with those of ACTIVERAG (with the Associate agent), with the highest accuracy achieved using Prompt 1. As shown in the table, ACTIVERAG outperformed all baseline models on NQ and TriviaQA, achieving top accuracy scores of 69.8% on NQ and 96.0% on TriviaQA with Modified Prompt 1. Remarkably, ACTIVERAG’s performance exceeded that of GPT-4 Turbo, demonstrating the effectiveness of its integration strategies. By leveraging these strategies, ACTIVERAG significantly boosted GPT-4 Mini’s accuracy, highlighting how its customized approach with the Associate agent can enhance LLM performance beyond traditional and advanced baselines.

**Table 4**:
| Model            | Best BLEU (NQ) | Best BLEU (TriviaQA) |
|------------------|----------------|-----------------------|
| Vanilla RAG      | 0.23           | 0.46                  |
| CoT              | 0.12           | 0.20                  |
| GPT-4.0 Mini     | 0.167          | 0.354                 |
| GPT-4 Turbo      | 0.187          | 0.443                 |
| ACTIVERAG (w. Associate Agent)      | 0.32           | 0.36                  |

Table 4 compares the best BLEU scores of baseline models with those of ACTIVERAG (with the Associate agent), with the best performance achieved using Prompt 1 for NQ and Prompt 2 for TriviaQA. The table shows that the ACTIVERAG achieved BLEU scores of 0.32 on NQ and 0.36 on TriviaQA, outperforming all other models on NQ, including GPT-4 Turbo (0.187). Although Vanilla RAG achieved a higher BLEU score on TriviaQA (0.46), ACTIVERAG's performance was still competitive and significantly better than CoT and GPT-4.0 Mini. These results highlight the effectiveness of the targeted prompt modifications, demonstrating superior response quality and alignment with reference answers, especially on NQ.

## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***




results:
nq - top10 - Roberta 
Accuracy	Average BLEU Score	 Cognition	Anchoring	Associate	Logician	CoT	    Epochs	  Learning Rate	  Loss Function  	Hidden Sizes
55.33%	    0.00771845063125037	     23	       47	       48	       25	     7	      10	       0.01	         MSELoss	      [128, 64]
55.33%	    0.00771845063125037	     23	       47	       48	       25	     7	      10	       0.001	     MSELoss	      [128, 64]
55.33%	    0.007710445651068163	 22	       47	       49	       25	     7	      10	       0.001	  CrossEntropyLoss	  [128, 64]
55.33%	    0.007710445651068163	 22	       47	       49	       25	     7	      15	       0.001	  CrossEntropyLoss	  [128, 64]

The data reveals that both configurations yielded an accuracy of 55.33% and a low average BLEU score of 0.0077, indicating moderate correctness in predictions but low textual similarity to reference answers. The agent selection shows a preference for Anchoring (47 times) and Associate (48 times), suggesting these agents provided more relevant responses compared to Cognition(23), Logician(25), and CoT(7). Despite using different learning rates (0.01 vs. 0.001), both setups with MSELoss and hidden layer sizes of [128, 64] achieved similar results, indicating that accuracy and BLEU scores were largely unaffected by this learning rate change.

