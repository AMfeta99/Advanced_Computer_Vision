# Brain Tumor Diagnosis
Brain tumor diagnostic app developed with Gradio. 

### Description
This app is based on *Vision Transformer (ViT)* model [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) that was fine-tuned for binary classification of brain X-rays, using the *dataset* [brain-tumor-dataset](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp) that contains 253 brain images. This dataset was originally created by Yousef Ghanem.

## Algorithm
The final version of the model is available at:
[vit-base-oxford-brain-tumor](https://huggingface.co/AMfeta99/vit-base-oxford-brain-tumor)

### Try the App Demo : [Link](https://huggingface.co/spaces/AMfeta99/brain_tumor_diagnosis)

![image](https://github.com/AMfeta99/Advanced_Computer_Vision/assets/74252797/227f5436-105d-4177-ae78-b42a5f69742a)



### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1
