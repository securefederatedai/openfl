# **Workflow Interface:** Fine-tuning neural-chat-7b-v3 using Intel(R) Extension for Transformers and OpenFL

## 1. About the dataset

We will be fine-tuning our model on the Medical Question Answering Dataset ([MedQuAD](https://github.com/abachaa/MedQuAD)). It is an open-source dataset comprised of medical question-answer pairs scrapped from various NIH websites.

## 2. About the model

Intel's [Neural-Chat-v3](https://huggingface.co/Intel/neural-chat-7b-v3) is a fine-tuned 7B parameter LLM from [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the open source dataset [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca).

Additional details in the fine-tuning can be found [here](https://medium.com/intel-analytics-software/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3).

## 3. Installing dependencies

In this tutorial, we will be fine-tuning Intel's neuralchat-7b model using OpenFL and Intel(R) Extension for Transformers

Start by installing Intel(R) Extension for Transformers (for stability, we will use v1.2.2) and OpenFL

```sh
pip install intel-extension-for-transformers==1.2.2
pip install openfl
```

From here, we can install requirements needed to run OpenFL's workflow interface and Intel(R) Extension for Transformer's Neural Chat framework

```sh
pip install -r requirements_neural_chat.txt
pip install -r requirements_workflow_interface.txt
```

## 4. Acquiring and preprocessing dataset

We can clone the dataset directly from the MedQuAD repository

```sh
git clone https://github.com/abachaa/MedQuAD.git
```

From here, we provide a preprocessing code to prepare the dataset to be readily ingestible by the fine-tuning pipeline

```sh
python preprocess_dataset.py
```

## 5: Running the tutorial

You are now ready to follow along in the tutorial notebook: `Workflow_Interface_NeuralChat.ipynb`

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