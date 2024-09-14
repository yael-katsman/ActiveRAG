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
The results of our experiments demonstrate that modifying the prompt strategies in the ActiveRAG framework led to notable improvements in performance on both the Natural Questions (NQ) and TriviaQA datasets. The original prompt provided solid performance, particularly achieving 95.0% Top-5 accuracy on TriviaQA and 68.0% Top-10 accuracy on NQ. By enhancing the prompt to focus on deeper contextual understanding (Modified Prompt 1), we saw an increase in Top-10 accuracy to 69.8% on NQ and a slight boost in BLEU scores from 0.28 to 0.32, showing better integration of contextual knowledge. Similarly, on TriviaQA, Modified Prompt 1 improved Top-10 accuracy to 96.0% with a corresponding increase in BLEU scores. Modified Prompt 2, which incorporated multi-perspective reasoning, further improved results on TriviaQA with the best Top-10 accuracy at 96.4% and BLEU scores of 0.34. However, Prompt 2 showed moderate gains on NQ, highlighting the complexity of finding the right balance between immediate retrieval accuracy and deeper reasoning. These findings underscore the potential of prompt engineering to refine knowledge integration and enhance both accuracy and contextual fluency in ActiveRAG.

## Discussion
Our experiments demonstrate that the different types of prompts used in the ActiveRAG framework play a critical role in enhancing the model's performance, particularly in tasks requiring deeper reasoning and contextual understanding. The Chain of Thought (CoT) prompt, which encourages step-by-step reasoning, provided a strong baseline by improving the model’s ability to break down complex queries and deliver more coherent responses.
The Anchoring prompt, focused on unfamiliar content extraction, proved especially useful in tasks where the model lacked knowledge. By retrieving and integrating unknown knowledge, the model’s responses were more informed and contextually aware, particularly in domains where it had limited prior knowledge.
The Associate prompt, designed to consolidate foundational and advanced knowledge, helped the model deepen its understanding of both basic and complex topics. This prompt was particularly effective in improving the model’s capacity to combine known information with newly retrieved data to provide more thorough answers.
The Logician prompt, which focuses on enhancing logical and causal reasoning, allowed the model to engage in more structured inference, improving its performance in tasks that required clear logical connections. This proved vital in queries involving cause-effect relationships or structured reasoning tasks.
Finally, the Cognition prompt, aimed at fact-checking and preventing model illusions, enabled the model to critically evaluate its output, preventing overconfidence in incorrect responses and enhancing its factual accuracy.

## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***
