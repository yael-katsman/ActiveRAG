# Bridging Retrieval and Reasoning: Enhancing the Associate Agent in ACTIVERAG for Knowledge-Intensive Tasks

## Project Outline
 [Overview](#overview)                             
 [Prerequisites](#prerequisites)                          
 [Installation](#installation)                      
 [File Descriptions](#file-descriptions)                         
 [Experiments](#experiments)
 [Prompts](#prompts)
 [Results](#results)                                    
 [Discussion](#discussion)                                      
 [Contact](#contact)

## Overview

ActiveRAG is an innovative Retrieval-Augmented Generation (RAG) framework that introduces an active learning mechanism to enhance the model’s understanding of external knowledge. Unlike traditional RAG systems that passively retrieve and integrate information, ActiveRAG uses Knowledge Construction to associate new information with previously acquired knowledge. This allows the model to refine and calibrate its internal understanding, improving reasoning and response quality.

Our project specifically focuses on enhancing the Associate agent within the ActiveRAG framework to boost performance in knowledge-intensive tasks. We evaluate this approach on datasets such as Natural Questions (NQ) and TriviaQA, using prompt engineering techniques like contextual extraction and multi-perspective reasoning to further improve model output.

## Prerequisites
To use ActiveRAG, you will need the following dependencies installed on your machine:

1. Python 3.8 or higher
2. PyTorch
3. Huggingface Transformers library
4. Other dependencies specified in requirements.txt

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

## File Descriptions

1. data - contains all the datasats: nq, popqa, triviaqa, webq.
2. src - ACTIVERAG source files: the Associate agent's original prompt can be found in src/prompt.py.
3. logs/log - contains JSON files documenting the detailed performance of the original ACTIVERAG on various datasets and retrieval settings (e.g., top-5 and top-10)
4. log_1 & log_2 - contain JSON files documenting the performance of ACTIVERAG with modified prompt 1 and prompt 2 respectively (can be found in......) on the triviaqa and nq datasets.

## Experiments
Models Used
1. ActiveRAG: Enhanced with multi-agent system, focusing on the Associate agent.
2. Vanilla RAG: Baseline retrieval-augmented generation.
3. Chain-of-Thought (CoT): Step-by-step reasoning to improve comprehension.
4. GPT-4.0 mini: Lightweight LLM for efficient inference.
5. GPT-4 Turbo: High-performance LLM used for comparison.

## Prompts
In our project, we utilize various prompt templates to improve the model’s reasoning and decision-making capabilities. Below are the different prompts used:

### 1. Chain of Thought (CoT) Prompt

This prompt encourages the model to solve problems through step-by-step reasoning.

```text
To solve the problem, please think and reason step by step, then answer.
question:
{question}

Generation Format:
Reasoning process:
Answer:
```

### 2. Anchoring Prompt
This prompt helps the model extract unfamiliar information from retrieved passages.
```text
You are a cognitive scientist. To answer the following question:
{question}
I will provide you with several retrieved passages:
Passages:
{passages}

Task Description:
Please extract content that may be unfamiliar to the model from these passages. This content should provide the model with relevant background knowledge, helping it better understand the question.
```

### 3. Associate Prompt
This prompt guides the model to consolidate foundational and advanced knowledge.
```text
You are a cognitive scientist. To answer the following question:
{question}
I will provide you with several retrieved passages:
Passages:
{passages}

Task Description:
Please extract foundational knowledge that may be familiar to the model, or advanced information beyond what the model already knows. Consolidate these contents to help the model deepen its understanding of the question.
```

### 4. Logician Prompt
This prompt improves the model's causal reasoning and logical inference abilities.
This prompt guides the model to consolidate foundational and advanced knowledge.
```text
You are a logician. To answer the following question:
{question}
I will provide you with several retrieved passages:
Passages:
{passages}

Task Description:
Please extract content from these passages that can enhance the model's causal reasoning and logical inference abilities. Consolidate these contents, and analyze how the selected information may impact the improvement of the model's reasoning capabilities.
```

### 5. Cognition Prompt
This prompt focuses on updating the model's knowledge to prevent factual errors and alleviate model illusions.
```text
You are a logician. To answer the following question:
You are a scientist researching fact-checking and model illusions in artificial intelligence. To answer the following question:
{question}
I will provide you with several retrieved passages:
Passages:
{passages}

Task Description:
Please extract content from these passages that may contradict the model's existing knowledge. Identify information that, when added, could update the model's knowledge and prevent factual errors.
```

## Results
We evaluated the impact of different prompts on the Associate agent's performance using Top-5 and Top-10 accuracy and BLEU scores on Natural Questions (NQ) and TriviaQA datasets.

1. Top-K Accuracy                                                    
   a. Original Prompt:
   Top-5 Accuracy: NQ - 66.8%, TriviaQA - 95.0%
   Top-10 Accuracy: NQ - 68.0%, TriviaQA - 95.4%
   Summary: Strong on factual retrieval but limited in handling complex queries, especially in NQ.
   
   b. Prompt 1 (Contextual Deepening):
   Top-5 Accuracy: NQ - 64.6%, TriviaQA - 94.8%
   Top-10 Accuracy: NQ - 69.8%, TriviaQA - 96.0%
   Summary: Improved Top-10 accuracy by deepening context, particularly in NQ.
   
   c. Prompt 2 (Multi-Perspective Reasoning):
   Top-5 Accuracy: NQ - 65.4%, TriviaQA - 95.2%
   Top-10 Accuracy: NQ - 68.0%, TriviaQA - 96.4%
   Summary: Best in complex fact retrieval for TriviaQA, stable for NQ.

2. BLEU Scores                 
   a. Original Prompt:
   Top-5 BLEU: NQ - 0.28, TriviaQA - 0.32                          
   b. Prompt 1:
   Top-5 BLEU: NQ - 0.30, TriviaQA - 0.34                        
   c. Prompt 2:
   Top-5 BLEU: NQ - 0.27, TriviaQA - 0.32

## Discussion
Our experiments show that prompt engineering plays a crucial role in improving the performance of the Associate agent within the ActiveRAG framework. The original prompt provided a solid baseline but struggled with deeper contextual reasoning, particularly in NQ tasks.

Prompt 1 successfully enhanced Top-10 accuracy by introducing more context-aware retrieval, showing improvements in both NQ and TriviaQA. This approach allowed the model to generate more contextually relevant answers, especially in tasks requiring nuanced understanding.

Prompt 2, focusing on multi-perspective reasoning, yielded the best results for TriviaQA, where combining facts from multiple sources was essential. While this prompt showed only moderate gains in NQ, it highlighted the potential of using expert-like reasoning to handle complex, fact-intensive questions.

Despite the improvements, there are opportunities for further refinement. Future work could explore dynamic prompt generation or incorporating feedback loops to improve real-time decision-making. Additionally, integrating few-shot learning might further boost performance in complex scenarios.
   
## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***
