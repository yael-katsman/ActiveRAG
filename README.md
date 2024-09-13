# Bridging Retrieval and Reasoning: Enhancing the Associate Agent in ACTIVERAG for Knowledge-Intensive Tasks

## Project Outline
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [File Descriptions](#file-descriptions)
- [Experiments](#experiments)
- [Results](#results)
- [Discussion](#discussion)
- [Contact](#contact)

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***

## Overview

ActiveRAG is designed to enhance traditional Retrieval-Augmented Generation (RAG) models by introducing an active learning approach. Unlike conventional RAG systems that passively integrate retrieved information, ActiveRAG employs a multi-agent system—including Anchoring, Logician, Cognition, and Associate agents—that actively engages with external knowledge. This system allows the model to refine and link new information with its existing knowledge base, improving the quality and contextual relevance of generated responses.

Building on ActiveRAG’s approach, our work specifically focuses on enhancing the Associate agent, which plays a crucial role in linking new data with the model’s existing knowledge. We ran ACTIVERAG using GPT-4.0 mini, rather than the original GPT-3.5 setup, to evaluate how prompt engineering techniques like contextual extraction and multi-perspective reasoning impact performance. We tested two modified prompts on the Natural Questions (NQ) and TriviaQA datasets, comparing ACTIVERAG’s results with baseline models like Vanilla RAG, Chain-of-Thought (CoT), and GPT-4 Turbo.

## Prerequisites
To use ActiveRAG, you will need the following dependencies installed on your machine:

1. Python 3.8 or higher
2. PyTorch
3. Huggingface Transformers library
4. Other dependencies specified in requirements.txt

## Installation
To reproduce the results of this project, you can clone the repository and install the required dependencies:

git clone [https://github.com/hillysegal1/ActiveRAG]                                                                                    
pip install -r requirements.txt

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



## Performance Breakdown



   
## Contact
For any questions or issues regarding this project, feel free to contact us at [ronshahar@campus.technion.ac.il], [hillysegal@campus.technion.ac.il], [yael-k@campus.technion.ac.il].
