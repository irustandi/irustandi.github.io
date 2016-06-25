---
layout: post
title: Performance comparison of Keras examples when run using Theano and Tensorflow
---

Setup: 2x Xeon E5 2670, 128GB RAM, Nvidia Geforce GTX 980 Ti (6GB), Ubuntu 14.04, Anaconda 4.0 running Python 2.7, Theano 0.8.2, Tensorflow 0.9.0

Each time is time for the first epoch. There is also quite a bit of
initialization time prior to the first epoch (in the order a few
seconds), but it is not carefully measured.

imdb_lstm.py

* Theano CPU: 214s
* Tensorflow CPU: 100s
* Theano GPU: 30s
* Tensorflow GPU: 153s

Note: Tensorflow GPU is actually SLOWER compared to Tensorflow CPU.

mnist_cnn.py

* Theano CPU: 223s
* Tensorflow CPU: 78s
* Theano GPU: 3s
* Tensorflow GPU: 24s
