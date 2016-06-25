---
layout: post
title: Performance comparison of Keras examples when run using Theano and TensorFlow
---

Setup: 2x Xeon E5 2670, 128GB RAM, Nvidia Geforce GTX 980 Ti (6GB), Ubuntu 14.04, Anaconda 4.0 running Python 2.7, Theano 0.8.2, TensorFlow 0.9.0

Each time is time for the first epoch. There is also quite a bit of
initialization time prior to the first epoch (in the order a few
seconds), but it is not carefully measured.

### imdb_lstm.py

* Theano CPU: 214s
* TensorFlow CPU: 100s
* Theano GPU: 30s
* TensorFlow GPU: 153s

Note: TensorFlow GPU is actually SLOWER compared to TensorFlow CPU.

### lstm_text_generation.py

* Theano CPU: around 6000s
* TensorFlow CPU: around 1200s
* Theano GPU: 213s
* TensorFlow GPU: 260s

### mnist_cnn.py

* Theano CPU: 223s
* TensorFlow CPU: 78s
* Theano GPU: 3s
* TensorFlow GPU: 24s
