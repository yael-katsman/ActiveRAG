# ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

Source code for our paper :  
***[ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547)***

If you find this work useful, please cite our paper  and give us a shining star 🌟

## Overview

ActiveRAG is designed to enhance traditional Retrieval-Augmented Generation (RAG) models by introducing an active learning approach. Unlike conventional RAG systems that passively integrate retrieved information, ActiveRAG employs a multi-agent system—including Anchoring, Logician, Cognition, and Associate agents—that actively engages with external knowledge. This system allows the model to refine and link new information with its existing knowledge base, improving the quality and contextual relevance of generated responses.

Building on ActiveRAG’s approach, our work specifically focuses on enhancing the Associate agent, which plays a crucial role in linking new data with the model’s existing knowledge. We ran ACTIVERAG using GPT-4.0 mini, rather than the original GPT-3.5 setup, to evaluate how prompt engineering techniques like contextual extraction and multi-perspective reasoning impact performance. We tested two modified prompts on the Natural Questions (NQ) and TriviaQA datasets, comparing ACTIVERAG’s results with baseline models like Vanilla RAG, Chain-of-Thought (CoT), and GPT-4 Turbo.

<p align="center">
  <img align="middle" src="fig/fig1.gif" style="max-width: 50%; height: auto;" alt="ActiveRAG"/>
</p>


## Quick Start

### Install from git

```bash
git clone https://github.com/OpenMatch/ActiveRAG
pip install -r requirements.txt
```

### Reproduction

We provide our request logs, so the results in the paper can be quickly reproduced:

```bash
python -m logs.eval --dataset nq --topk 5
```

**Parameters:**

- `dataset`: dataset name.
- `topk`: using top-k of retrieved passages to augment.

##  Re-request

We also provide the full request code, you can re-request for further exploration.

First, set your own api-key in agent file:

```python
openai.api_key = 'sk-<your-api-key>'
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

## Citation
```
@article{xu2024activerag,
  title={ActiveRAG: Revealing the Treasures of Knowledge via Active Learning},
  author={Xu, Zhipeng and Liu, Zhenghao and Liu, Yibin and Xiong, Chenyan and Yan, Yukun and Wang, Shuo and Yu, Shi and Liu, Zhiyuan and Yu, Ge},
  journal={arXiv preprint arXiv:2402.13547},
  year={2024}
}
```

## Contact Us

If you have questions, suggestions, and bug reports, please send a email to us, we will try our best to help you. 

```bash
xuzhipeng@stumail.neu.edu.cn  
```

