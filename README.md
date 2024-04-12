# Cassava-Leaf-Disease-Classification

This is not how we properly write README.md files, but please forgive it.

The goal of the competition was to train a model capable of classifying images of cassava leaves with certain types of diseases.

I trained a very simple Convolutional Neural Network for this in 200 epochs. I didn't submit a transfer learning-based model, but rather a CNN from scratch yet it achieved a test data private score of 0.8315. The top scorer is 0.9132. The architecture is only 7 conv layers deep with data augmentation.

One thing I found interesting. I initially started using GPUs. When I started fitting the model, the RAM shoots up. To solve this memory constraint, I divided the train dataset into multiple sub-dataset objects, and fit the model one sub-dataset object at a time. After fitting all the sub-dataset objects, I considered it as one epoch. The model did not overfit. However, when I got the chance to use a TPU, the RAM GB was pretty high enough to prevent this memory constraint. So I fit the model to the entire dataset, but the model, surprisingly, was overfitting. I made sure everything was constant and unchanged; I only changed the accelerator and how the dataset object is fed to the model training. I think the difference has something to do with calling model.fit() on every sub-dataset object.
