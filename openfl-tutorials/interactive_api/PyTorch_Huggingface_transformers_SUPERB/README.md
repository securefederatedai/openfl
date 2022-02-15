# Hugging face transformers tutorial for audio classification using PyTorch

Transformers have been a driving point for breakthrough developments in the Audio and Speech processing domain. Recently, Hugging Face dropped the State-of-the-art Natural Language Processing library Transformers v4.30 and extended its reach to Speech Recognition by adding one of the leading Automatic Speech Recognition models by Facebook called the Wav2Vec2.

### About model: Wav2Vec2

This tutorial uses [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#wav2vec2forsequenceclassification) model which is a speech model checkpoint from the [Model Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads). The Wav2Vec2 model was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) which shows that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. We will fine-tune this pretrained speech model for Automatic Speech Recognition in this tutorial.

### About dataset: Keyword spotting (KS) from SUPERB

Keyword spotting subset from [SUPERB](https://huggingface.co/datasets/superb) dataset is used. Keyword Spotting (KS) detects preregistered keywords by classifying utterances into a predefined set of words. The dataset consists of ten classes of keywords, a class for silence, and an unknown class to include the false positive. The evaluation metric is accuracy (ACC).

### Links

* [Huggingface transformers on Github](https://github.com/huggingface/transformers)
* [Original Huggingface notebook audio classification example on Github](https://github.com/huggingface/notebooks/blob/master/examples/audio_classification.ipynb)

### How to run this tutorial (without TLS and locally as a simulation):

Go to example [folder](./)

```sh
export PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB=<openfl_folder>/openfl-tutorials/interactive_api/PyTorch_Huggingface_transformers_SUPERB
```

1. Run director:

```sh
cd $PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB/director
bash start_director.sh
```

2. Run envoy:

```sh
cd $PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB/envoy
pip install -r sd_requirements.txt
bash start_envoy.sh
```

Optional: start second envoy:

- Copy `$PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB/envoy` to another folder, change the config and envoy name in
  start_envoy.sh and run from there:

```sh
cd $PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB/envoy_two
bash start_envoy.sh
```

3. Run `PyTorch_Huggingface_transformers_SUPERB.ipynb` jupyter notebook:

```sh
cd $PYTORCH_HUGGINGFACE_TRANSFORMERS_SUPERB/workspace
jupyter lab PyTorch_Huggingface_transformers_SUPERB.ipynb
```
