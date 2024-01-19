# **Workflow Interface:** Fine-tuning neural-chat-7b-v3 using Intel(R) Extension for Transformers and OpenFL

## 1. About the dataset

We will be fine-tuning our model on the Medical Question Answering Dataset ([MedQuAD](https://github.com/abachaa/MedQuAD)). It is an open-source dataset comprised of medical question-answer pairs scrapped from various NIH websites.

## 2. About the model

Intel's [Neural-Chat-v3](https://huggingface.co/Intel/neural-chat-7b-v3) is a fine-tuned 7B parameter LLM from [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the open source dataset [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca).

Additional details in the fine-tuning can be found [here](https://medium.com/intel-analytics-software/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3).

## 3. Running the tutorial

Follow along step-by-step in the [notebook](Workflow_Interface_NeuralChat.ipynb) to learn how to fine-tune neural-chat-7b on the MedQuAD dataset

## Reference:
```
@ARTICLE{BenAbacha-BMC-2019,    
          author    = {Asma {Ben Abacha} and Dina Demner{-}Fushman},
          title     = {A Question-Entailment Approach to Question Answering},
          journal = {{BMC} Bioinform.}, 
          volume    = {20},
          number    = {1},
          pages     = {511:1--511:23},
          year      = {2019},
          url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4}
           } 
```