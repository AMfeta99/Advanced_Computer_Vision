# Brain Tumor Diagnosis

![brain-tumor-scansjpg__1024x480_q100_crop_subsampling-2_upscale](https://github.com/AMfeta99/Advanced_Computer_Vision/assets/74252797/cf675a49-d0d0-45f1-98eb-37b904dca82b)


### Description
This app is based on *Vision Transformer (ViT)* model [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224).

ViT model is originaly a transformer encoder model pre-trained and fine-tuned on ImageNet 2012. It was introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. The model processes images as sequences of 16x16 patches, adding a [CLS] token for classification tasks, and uses absolute position embeddings. Pre-training enables the model to learn rich image representations, which can be leveraged for downstream tasks by adding a linear classifier on top of the [CLS] token. The weights were converted from the timm repository by Ross Wightman.


## Algorithm & Training Process
ViT was fine-tuned for binary classification of brain X-rays, using the *dataset* [brain-tumor-dataset](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp) that contains 253 brain images. This dataset was originally created by Yousef Ghanem.

The original dataset was splitted into training and evaluation subsets, 80% for training and 20% for evaluation. For robust framework evaluation, the evaluation subset is further split into two equal parts for validation and testing. This results in three distinct datasets: training, validation, and testing. All pre-processing steps and training process can be found on [Notebook](https://github.com/AMfeta99/Advanced_Computer_Vision/blob/main/Computer_Vision_HF/brain_tumor_diagnosis_app_HF/Transfer_learning_image_classification.ipynb)

The final version of the model is available at:
[vit-base-oxford-brain-tumor](https://huggingface.co/AMfeta99/vit-base-oxford-brain-tumor)

## Application Demo
Brain tumor diagnostic app was developed with Gradio. This is hosted by Hugging Face Spaces, so anyone can have access to it.
### Try the Demo : [App](https://huggingface.co/spaces/AMfeta99/brain_tumor_diagnosis)
[x-ray sample available on this repo]

![image](https://github.com/AMfeta99/Advanced_Computer_Vision/assets/74252797/227f5436-105d-4177-ae78-b42a5f69742a)


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1
