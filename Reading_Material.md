# Paper List
Paper reading list in LLMs: 
- [Efficient Large Language Model](#Efficient-Large-Language-Model)
- [Understand LLMs](#Understand-LLMs)
  - [Model Structure](#model-structure)
  - [Gradient Approximate](#gradient-approximate)
  - [CoT Interpretability](#cot-interpretability)
  - [LLM as latent variable Model](#LLM-as-latent-variable-Model)
  - [Training Data](#training-data)
  - [Emergence](#emergence)
- [Reasoning in LLMs](#Reasoning-in-LLMs)
- [Causal Inference](#Causal-Inference)
- [Cognitive LLMs](#cognitive-llm)
- [Model Editing](#model-edit)
- [Safe LLMs](#safe-llm)
- [Alignment LLMs](#alignment-llm)

This repository will keep updating ... 🤗
***


## Efficient Large Language Model
* Fast inference from transformers via speculative decoding. ICLR23 oral. [paper](https://arxiv.org/abs/2211.17192)
* SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification. [paper](https://arxiv.org/pdf/2305.09781.pdf)

👆 [Back to Top](#paper-list)

## Understand LLMs
**Tracr**: TRAnsformer Compiler for RASP

### Model Structure
[My own Notes](https://zhuanlan.zhihu.com/p/652269984)
* In-context Learning and Induction Heads (Anthropic AI) [paper](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* Transformers as Algorithms: Generalization and Stability in In-context Learning. ICML23. [paper](https://arxiv.org/pdf/2301.07067.pdf)
  - Prerequisite reading
    - A Mathematical Framework for Transformer Circuits. [paper](https://transformer-circuits.pub/2021/framework/index.html#three-kinds-of-composition)
### Gradient Approximate
(machine learning/theory)
* Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers2022. [paper](https://arxiv.org/abs/2212.10559)
* What LEARNING ALGORITHM IS IN-CONTEXT LEARNING? INVESTIGATIONS WITH LINEAR MODELS. ICLR23. [paper](https://arxiv.org/abs/2211.15661)
* Transformers Learn In-Context by Gradient Descent. ICML23. [paper](https://arxiv.org/abs/2212.0767)

### CoT Interpretability
* Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective. [paper](https://arxiv.org/abs/2305.15408)

### LLM as latent variable Model
* An Explanation of In-context Learning as Implicit Bayesian Inference. [paper](https://arxiv.org/abs/2111.02080)
* Schema-learning and rebinding as mechanisms of in-context learning and emergence. DeepMind, June23. [paper](https://arxiv.org/pdf/2307.01201.pdf)
  - Propose a sequence learning model based on action->latent variable->observed variable Generation process.
### Training Data
- Distribution
  * Data Distributional Properties Drive Emergent In-Context Learning in Transformers. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf)
- Diversity
  * Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression​. [paper](proceedings.neurips.cc/paper_files/paper/2022/file/77c6ccacfd9962e2307fc64680fc5ace-Paper-Conference.pdf)
  * 
### Emergence
* A Theory for Emergence of Complex Skills in Language Models. [paper](​https://arxiv.org/pdf/2307.15936.pdf)

👆 [Back to Top](#paper-list)
## Causal Inference
* Causal interventions expose implicit situation models for commonsense language understanding. ACL-findings23. [paper](https://arxiv.org/pdf/2306.03882.pdf)
* 

👆 [Back to Top](#paper-list)
## Reasoning in LLMs
* Make a Choice! Knowledge Base Question Answering with In-Context Learning
* STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning.  Neurips22, cite110+. [paper](https://​openreview.net/forum?id=_3ELRdg2sgI)
* ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models.[paper](https://arxiv.org/pdf/2303.16421.pdf)

👆 [Back to Top](#paper-list)
## Cognitive LLMs
* Dissociating Language and thought in large Language Model: a cognitive perspective. [paper](https://arxiv.org/pdf/2301.06627.pdf)
* Using cognitive psychology to understand GPT-3. [paper](https://arxiv.org/abs/2206.14576) 
👆 [Back to Top](#paper-list)

## Model Editing
* Can We Edit Factual Knowledge by In-Context Learning? [paper](https://arxiv.org/pdf/2305.12740.pdf)
👆 [Back to Top](#paper-list)

## Safe/Reliable LLM
Align the LLM to human values to avoid risks by either fine-tuning or frozen-llm.
* Do the rewards justify the means? measuring trade-offs between rewards and ethical behavior in the machiavelli benchmark. ICML23, oral.[paper](https://arxiv.org/pdf/2304.03279.pdf)

  TL;DR: A benchmark for reward and ethical behavior evaluation. And Two baseline models.
  
* Fine-tuning Language Models for Factuality. Arxiv.2023.11. Stanford Uni. [paper](https://arxiv.org/pdf/2311.08401.pdf)

  TL;DR: They fine-tune the model on annotation data which are selected as factually correct sampled from the model responses.
  
👆 [Back to Top](#paper-list)

## Alignment LLM
Comparing to the last entry, its applications are more general.

(1) Fine-tune/Training on curated datasets. The keypoints could be the datasets.

* LIMA: Less Is More for Alignmen. 2023.05. MetaAI/CMU [paper](https://arxiv.org/pdf/2305.11206.pdf)
* Fine-Grained Human Feedback Gives Better Rewards for Language Model Training. Neurips23, Oral [paper](https://arxiv.org/abs/2306.01693)
* Learning Customized Human Preferences. 2023.09. Arxiv. [Github](https://github.com/Linear95/DSP)
  TL;DR: To learn a customized reward model well while preserving its general preference ability. This is a Domain-Specific Preference (DSP) dataset and test multiple training and data strategies for customized Reward Model learning.
(2) Parameter-Frozen

* RAIN: Your Language Models Can Align Themselves without Finetuning. Peking Uni. 202309. [Github](https://github.com/SafeAILab/RAIN) [paper](https://arxiv.org/abs/2309.07124)

  TL;DR: Beam Search similarily
  
👆 [Back to Top](#paper-list)

