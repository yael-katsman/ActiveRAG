# ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***

If you find this work useful, please cite our paper  and give us a shining star 🌟

## Overview

ActiveRAG is designed to enhance traditional Retrieval-Augmented Generation (RAG) models by introducing an active learning approach. Unlike conventional RAG systems that passively integrate retrieved information, ActiveRAG employs a multi-agent system—including Anchoring, Logician, Cognition, and Associate agents—that actively engages with external knowledge. This system allows the model to refine and link new information with its existing knowledge base, improving the quality and contextual relevance of generated responses.

Building on ActiveRAG’s approach, our work specifically focuses on enhancing the Associate agent, which plays a crucial role in linking new data with the model’s existing knowledge. We ran ACTIVERAG using GPT-4.0 mini, rather than the original GPT-3.5 setup, to evaluate how prompt engineering techniques like contextual extraction and multi-perspective reasoning impact performance. We tested two modified prompts on the Natural Questions (NQ) and TriviaQA datasets, comparing ACTIVERAG’s results with baseline models like Vanilla RAG, Chain-of-Thought (CoT), and GPT-4 Turbo.

