# Bridging Retrieval and Reasoning: Enhancing the Associate Agent in ACTIVERAG for Knowledge-Intensive Tasks

## Project Outline
 [Overview](#overview)                             
 [Prerequisites](#prerequisites)                          
 [Setup Guide](#setup-guide)                      
 [File Descriptions](#file-descriptions)                         
 [Experiments](#experiments)
 [Prompts](#prompts)
 [Results](#results)                                    
 [Discussion](#discussion)                                      
 [Contact](#contact)

## Overview

ActiveRAG is an innovative Retrieval-Augmented Generation (RAG) framework that introduces an active learning mechanism to enhance the model’s understanding of external knowledge. Unlike traditional RAG systems that passively retrieve and integrate information, ActiveRAG uses Knowledge Construction to associate new information with previously acquired knowledge. This allows the model to refine and calibrate its internal understanding, improving reasoning and response quality.

Our project specifically focuses on enhancing the Associate agent within the ActiveRAG framework to boost performance in knowledge-intensive tasks. We created two modifications of the Associate agent's original prompt  and evaluated our approach on datasets such as Natural Questions (NQ) and TriviaQA, using prompt engineering techniques like contextual extraction and multi-perspective reasoning to further improve model output.

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

## Results
The results of our experiments demonstrate that modifying the prompt strategies in the ActiveRAG framework led to notable improvements in performance on both the Natural Questions (NQ) and TriviaQA datasets. The original prompt provided solid performance, particularly achieving 95.0% Top-5 accuracy on TriviaQA and 68.0% Top-10 accuracy on NQ. By enhancing the prompt to focus on deeper contextual understanding (Modified Prompt 1), we saw an increase in Top-10 accuracy to 69.8% on NQ and a slight boost in BLEU scores from 0.28 to 0.32, showing better integration of contextual knowledge. Similarly, on TriviaQA, Modified Prompt 1 improved Top-10 accuracy to 96.0% with a corresponding increase in BLEU scores. Modified Prompt 2, which incorporated multi-perspective reasoning, further improved results on TriviaQA with the best Top-10 accuracy at 96.4% and BLEU scores of 0.34. However, Prompt 2 showed moderate gains on NQ, highlighting the complexity of finding the right balance between immediate retrieval accuracy and deeper reasoning. These findings underscore the potential of prompt engineering to refine knowledge integration and enhance both accuracy and contextual fluency in ActiveRAG.

## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***
