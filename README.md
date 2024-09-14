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

``` git clone [https://github.com/hillysegal1/ActiveRAG] ```                                                                                
``` pip install -r requirements.txt ```

### Reproduction
You can reproduce the results from our paper using the following command:           
``` python -m logs.eval --dataset nq --topk 5 ```


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
We explored different prompt engineering techniques to enhance the Associate agent within the ActiveRAG framework. Each prompt was designed to tackle specific challenges in knowledge retrieval and reasoning:

1. Original Prompt                         
The baseline prompt, focused on retrieving top-k relevant passages without additional reasoning steps.
Strengths: Good at handling straightforward factual questions.
Limitations: Struggled with complex queries that required deeper reasoning or integration of multiple pieces of knowledge.
2. Prompt 1: Contextual Deepening                      
Designed to improve contextual awareness by linking retrieved passages more closely to the query.
Purpose: To enhance the model’s understanding of nuanced relationships between the query and retrieved information.
Strengths: Improved performance in Top-10 accuracy for NQ by providing more contextually relevant answers.
Limitations: Slight drop in Top-5 accuracy as the model spent more time deepening context rather than focusing on direct retrieval.
3. Prompt 2: Multi-Perspective Reasoning                        
Focused on engaging the model in reasoning from multiple perspectives (e.g., logical reasoning, factual retrieval).
Purpose: To synthesize information from various sources and produce a more comprehensive response.
Strengths: Showed strong performance in TriviaQA, where combining facts from multiple retrieved passages is essential.
Limitations: Did not significantly improve performance in NQ, where deeper contextual reasoning was more effective.


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
