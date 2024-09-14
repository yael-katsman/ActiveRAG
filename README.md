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

Building on ActiveRAG’s approach, our work specifically focuses on enhancing the Associate agent, which plays a crucial role in linking new data with the model’s existing knowledge. We ran ACTIVERAG using GPT-4.0 mini, rather than the original GPT-3.5 setup, to evaluate how prompt engineering techniques like contextual extraction and multi-perspective reasoning impact performance. We tested two modified prompts on the Natural Questions (NQ) and TriviaQA datasets, comparing ACTIVERAG’s results with baseline models, specifially, Vanilla RAG and Chain-of-Thought (CoT). To further evaluate our modifications, we also compared our results against GPT-4 turbo. 

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
