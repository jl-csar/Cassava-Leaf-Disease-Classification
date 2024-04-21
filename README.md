# Cassava-Leaf-Disease-Classification

### Badges:

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-violet.svg)](https://opensource.org/licenses/Apache-2.0)

<br>

### Description

The goal of the competition is to train a model capable of classifying images of cassava leaves with certain types of diseases. As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. With AI, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

The dataset and competition is from Kaggle: [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) 

<br>

### Table of Contents

- [Model Architecture](https://github.com/jl-csar/Cassava-Leaf-Disease-Classification/blob/main/README.md#model-architecture)
- [Results](https://github.com/jl-csar/Cassava-Leaf-Disease-Classification/blob/main/README.md#results)
- [Notes](https://github.com/jl-csar/Cassava-Leaf-Disease-Classification/blob/main/README.md#notes)
- [License](https://github.com/jl-csar/Cassava-Leaf-Disease-Classification/blob/main/README.md#license)

<br>

### Model Architecture

I trained a very simple Convolutional Neural Network in 200 epochs. The CNN was 7 conv layers deep with data augmentation and dropout. The dropout, which had a rate of 0.2, was done alternately up until the dense layers. Two data augmentation techniques were made which are random contrast and random rotation. The model had 237,573 trainable parameters. The optimizer used was Adam.

I tried lots of hyperparameter tuning but came upon with this architecture. I do think that data augmentation helped the model learn more features. Coupling it with the regularizing dropout helps prevent the model from overfitting. I didn't submit a transfer learning-based model yet it achieved a test data private score of 0.8315. The top scorer is 0.9132.

I initially started using GPUs (T4 x2) with 30 GB of RAM. When I started fitting the model, the RAM shoots up; it needs around 130 GB of RAM (I didn't use generators, only tensorflow datasets). To solve this memory constraint, I divided the train dataset into multiple sub-dataset objects, and fit the model one sub-dataset object at a time. After fitting all the sub-dataset objects, I considered it as one epoch. When I got the chance to use a TPU (VM v3-8), the RAM GB (330 GB) was high enough to prevent this memory constraint.

<br>

### Results

As early as 100-125 epochs, the model started to plateau around a validation accuracy of 0.81. This is most probably due to the model not being flexible or complex enough with only 237,573 trainable parameters, however, I didn't try to increase the layers or parameters and see the difference in improvement. The training and validation loss and accuracy were not oscillating and were pretty stable.

<br>

### Notes

1. Filenames and contents
    - training.ipynb: the training notebook that used TPUs
    - training_memory_constraints_dataset_objects.ipynb: the training notebook that used GPUs
    - training_transfer_learning.ipynb: the training notebook where transfer learning was used using ResNet-50
    - training_submission.ipynb: the submitted notebook

<br>

### License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

The Apache License 2.0 is a permissive open-source license that allows you to use, modify, and distribute the software for any purpose, including commercial purposes, as long as you satisfy the conditions of the license.

Copyright 2024 Julius Ceasar Dumaslan. All rights reserved.
