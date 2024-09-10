# Vision Transformer for Image Classification
## Overview
This project focuses on implementing a Vision Transformer (ViT) for image classification. The ViT model divides images into fixed-size patches and processes them using transformer blocks to capture relationships between patches. The final classification is performed using a fully connected layer. This architecture provides an alternative to traditional CNNs by leveraging the power of transformers to model global dependencies in the image data.

## Model Implementation
### Patch Embedding
- Images are divided into 16x16 patches.
- A convolutional layer with stride and kernel size equal to the patch size generates the patch embeddings. For this project, the embedding size was set to 384.
- A learnable class token is appended to the patch embeddings.
- Position embeddings are added to the patches to capture the spatial information.
  
### Transformer Block
- The Transformer block consists of 6 transformer layers.
- Each layer includes Layer Normalization, Multi-Head Self-Attention (with 64 heads), and a Multi-Layer Perceptron (MLP) with skip connections.
- The Multi-Head Attention mechanism computes relationships between patches using Query, Key, and Value embeddings, normalized via softmax.

### Classification Head
- The class token output from the final transformer block is passed through the classification head, resulting in 5-class logits.

## Dataset
We first download the JSON annotation file for MS-COCO dataset from the [official COCO website](https://cocodataset.org/#download). Then using the COCO API, we load 2000 images each for the above mentioned classes and resize them into (64 × 64) and save 1500 of them into the training dataset and 500 of them into the test dataset.

## Training
- Optimizer: Adam with beta values (0.9, 0.99).
- Learning Rate: 0.0001.
- Epochs: 10.
- Batch Size: 32.
- Loss Function: Binary Cross Entropy Loss.

## Requirements
Please refer to the attached PDF file for further details on the outputs' implementation and visual representation and details of all the libraries and datasets required. 
Also, please check the imports in the .ipynb or .py file.
