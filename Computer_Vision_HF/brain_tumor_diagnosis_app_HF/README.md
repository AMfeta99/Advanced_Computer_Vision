# Brain Tumor Diagnosis

![brain-tumor-scansjpg__1024x480_q100_crop_subsampling-2_upscale](https://github.com/AMfeta99/Advanced_Computer_Vision/assets/74252797/cf675a49-d0d0-45f1-98eb-37b904dca82b)

### Introduction & literature
The human brain, a crucial organ located in the cranium, manages various functions through a complex network of billions of neurons coordinating electrical and chemical impulses. This organ is central to perception, emotion, and character, with distinct components like the cerebral cortex, responsible for consciousness, and the cerebellum, which handles balance and coordination. Despite its resilience, the brain can develop tumors, which are abnormal growths of cells.

The incidence rate of brain and other central nervous system (CNS) tumors varies globally, with the highest rates observed in North America, Northern Europe, and Australia. In the United States, the annual incidence rate is approximately 23 per 100,000 people. For children, brain tumors are the second most common type of cancer, with about 4.7 new cases per 100,000 children annually.

Brain tumors, characterized by the rapid development of abnormal brain cells, pose a major health risk due to their potential to severely impair organ function or cause death. Diagnosing brain tumors typically involves imaging tests and biopsies. Detecting these tumors in MRI scans manually is challenging and time-consuming, often leading to inaccuracies.

Advances in neuro-oncology, particularly with deep learning and artificial intelligence (AI), have significantly improved the analysis of medical images and consequently improved patient care outcomes. AI assists in the detection, diagnosis, and characterization of various medical conditions, enhancing early diagnosis and treatment decisions. Transfer learning from pre-existing models has been shown to improve the efficiency and accuracy of medical image analysis, significantly reducing training time and computational costs. 

Models such as VGG, ResNet, Inception, MobileNet, DenseNet, and recently YOLO have already been tested in the literature, demonstrating remarkable effectiveness in identifying intricate patterns in medical images, particularly in cancer detection.

All the previous models are convolutional neural network (CNN) architectures. In this study, I have fine-tuned a transformer-based model, Vision Transformer (ViT), achieving a performance of 92.31% accuracy on the test dataset. Transformer-based models are a different class of models that use self-attention mechanisms rather than convolutional layers for image analysis.

### Description
This app is based on *Vision Transformer (ViT)* model [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224).

ViT model is originaly a transformer encoder model pre-trained and fine-tuned on ImageNet 2012. It was introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. The model processes images as sequences of 16x16 patches, adding a [CLS] token for classification tasks, and uses absolute position embeddings. Pre-training enables the model to learn rich image representations, which can be leveraged for downstream tasks by adding a linear classifier on top of the [CLS] token. The weights were converted from the timm repository by Ross Wightman.


## Algorithm & Training Process
ViT was fine-tuned for binary classification of brain MRI scans, using the *dataset* [brain-tumor-dataset](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp) that contains 253 brain images. This dataset was originally created by Yousef Ghanem.

The original dataset was splitted into training and evaluation subsets, 80% for training and 20% for evaluation. For robust framework evaluation, the evaluation subset is further split into two equal parts for validation and testing. This results in three distinct datasets: training, validation, and testing. 

The model's performance is enhanced using image enhancement and data augmentation techniques. All pre-processing steps and training process can be found on [Notebook](https://github.com/AMfeta99/Advanced_Computer_Vision/blob/main/Computer_Vision_HF/brain_tumor_diagnosis_app_HF/Transfer_learning_image_classification.ipynb)

The final version of the model is available at:
[vit-base-oxford-brain-tumor](https://huggingface.co/AMfeta99/vit-base-oxford-brain-tumor)

## Application Demo
Brain tumor diagnostic app was developed with Gradio. This is hosted by Hugging Face Spaces, so anyone can have access to it.
### Try the Demo : [App](https://huggingface.co/spaces/AMfeta99/brain_tumor_diagnosis)


![image](https://github.com/AMfeta99/Advanced_Computer_Vision/assets/74252797/8bead0a4-e3f7-4db3-819f-30a70c55c8f7)


### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| 0.6519        | 1.0   | 11   | 0.3817          | 0.8      | 0.8476    | 0.8    | 0.7751 |
| 0.2616        | 2.0   | 22   | 0.0675          | 0.96     | 0.9624    | 0.96   | 0.9594 |
| 0.1219        | 3.0   | 33   | 0.1770          | 0.92     | 0.9289    | 0.92   | 0.9174 |
| 0.0527        | 4.0   | 44   | 0.0234          | 1.0      | 1.0       | 1.0    | 1.0    |

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 20
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4


### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1

### References
- Abdusalomov, A. B., Mukhiddinov, M., & Whangbo, T. K. (2023). Brain Tumor Detection Based on Deep Learning Approaches and Magnetic Resonance Imaging. Cancers, 15(16), 4172. https://doi.org/10.3390/cancers15164172
- Mathivanan, S.K., Sonaimuthu, S., Murugesan, S. et al. Employing deep learning and transfer learning for accurate brain tumor detection. Sci Rep 14, 7232 (2024). https://doi.org/10.1038/s41598-024-57970-7
- American Cancer Society. (2023). Key statistics for brain and spinal cord tumors. Retrieved from https://www.cancer.org/cancer/types/brain-spinal-cord-tumors-adults/about/key-statistics.html
- National Cancer Institute. (2023). Adult brain tumor treatment (PDQ®)–Health professional version. Retrieved from https://www.cancer.gov/types/brain/hp/adult-brain-treatment-pdq
- [Dataset](https://universe.roboflow.com/roboflow-100/brain-tumor-m2pbp)
- [Hugging_Face](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)
