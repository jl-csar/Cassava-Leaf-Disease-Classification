# Cassava-Leaf-Disease-Classification

### Badges:

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<br>

### Description

The goal of the competition is to train a model capable of classifying images of cassava leaves with certain types of diseases. As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. With AI, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

The dataset and competition is from Kaggle: [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) 

<br>

### Architecture

I trained a very simple Convolutional Neural Network in 200 epochs. The CNN was 7 conv layers deep with data augmentation and dropout. The dropout, which had a rate of 0.2, was done alternately up until the dense layers. Two data augmentation techniques were made which are random contrast and random rotation. The model had 237,573 trainable parameters. The optimizer used was Adam.

I tried lots of hyperparameter tuning but came upon with this architecture. I do think that data augmentation helped the model learn more features. Coupling it with the regularizing dropout helps prevent the model from overfitting. I didn't submit a transfer learning-based model yet it achieved a test data private score of 0.8315. The top scorer is 0.9132. 

One thing I found interesting. I initially started using GPUs. When I started fitting the model, the RAM shoots up. To solve this memory constraint, I divided the train dataset into multiple sub-dataset objects, and fit the model one sub-dataset object at a time. After fitting all the sub-dataset objects, I considered it as one epoch. The model did not overfit. However, when I got the chance to use a TPU, the RAM GB was pretty high enough to prevent this memory constraint. So I fit the model to the entire dataset, but the model, surprisingly, was overfitting. I made sure everything was constant and unchanged; I only changed the accelerator and how the dataset object is fed to the model training. I think the difference has something to do with calling model.fit() on every sub-dataset object.
