# ActiveRAG

Source code for our paper :  
***ActiveRAG: Revealing the Treasures of Knowledge via Active Learning***

If you find this work useful, please cite our paper  and give us a shining star 🌟

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



## Contact Us

If you have questions, suggestions, and bug reports, please send a email to us, we will try our best to help you. 

```bash
xuzhipeng@stumail.neu.edu.cn  
```

