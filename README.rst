A tensorflow backend to spinnaker
=================================


Goal
----

The goal of this dissertation is to see, whether spinnaker_
is a competitive target for the training of neural
networks (NNs), compared to state of the art approaches,
like using accelerators (GPGPUs_ or TPUs_) or distributed
learning (see e.g. `distributed learning with keras
<https://www.tensorflow.org/tutorials/distribute/keras?hl=es-419>`_).

For this I propose the following benchmark for image
recognition on the mnist_ data set:

1. train the following neural networks [1]_ on spinnaker
   and compare the raw speed of training to state of the
   art approaches for training (to be determined which):

   * 1-hidden-layer fully connected NN

   * 2-hidden-layer fully connected NN

   * LeNet-1

   * LeNet-4

   * LeNet-5

   * Boosted LeNet-4

2. while the first benchmark would give information on the
   performance of spinnaker, I personally believe,
   comparing a technology that can scale to the size of a
   supercomputer against unscalable accelerated
   architectures could be misleading.
   Therfore, I would like to use the same benchmark to
   analyze the energy efficiency of spinnaker as well, to
   see whether spinnaker can be a more scalable and more
   cost effective in terms of energy consumption.

.. [1] For more information see `here <https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17>`_


Why the better title would be: a keras backend to spinnaker
-----------------------------------------------------------

I'm making the daring assumption, that implementing a new
backend for tensorflow is not possible for one person in
three months, without prior knowledge of
`XLA <https://www.tensorflow.org/xla>`_ (implementing an
XLA backend is significantly simpler than retargeting the
tensorflow operations directly, according to [2]_),
`LLVM <http://llvm.org>`_ and C++ (regretably, I have no
experiences with any of these technologies).
Especially, since spinnaker does not provide a target for
llvm directly (third scenario from [2]_, the one requiring
most work).

It is even a question, if XLA can be interfaced properly
with the spinnaker runtime, which rather significantly
defers from classical XLA targets, like CPUs and GPUs.

Therefore, rather than interfacing spinnaker with the low
level tensorflow computational graph, I propose interfacing
with the conceptual graph provided by the deep learning
library `Keras <https://keras.io>`_, the de facto standard
frontend for tensorflow, when it comes to implementing deep
learning models.

According to the goals of this dissertation, I feel this is
the much more sensible approach, especially considering,
that tensorflow is not at all the kind of environment that
we want to port to spinnaker (tensorflow abstracts linear
algebra operations as a graph, which can be optimized and
distributed onto clusters and accelerators
(see [TF2015]_), resulting in a very complex runtime.
Overhead, which we want to avoid in the heterogenous
setting spinnaker provides).

.. [2] `Developing a new backend for XLA <https://www.tensorflow.org/xla/developing_new_backend>`_

.. [TF2015] `TensorFlow:Large-Scale Machine Learning on Heterogeneous Distributed Systems <http://download.tensorflow.org/paper/whitepaper2015.pdf>`_


Proposed API
------------

.. code:: python

   import tensorflow as tf

   import keras2spinnaker as k2s

   # load mnist data set
   mnist = tf.keras.datasets.mnist
   (X_train, y_train), (X_test, y_test) = mnist.load_data()

   # normalize rgb colors of the images
   X_train, X_test = X_train / 255.0, X_test / 255.0

   # create neural net
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile( optimizer='adam'
                , loss='sparse_categorical_crossentropy'
                , metrics=['accuracy'] )

   # convert and run the model as spinnaker graph and
   # update the keras model with the weights
   k2s.fit(model, X_train, y_train)

   # runs on local machine
   s = model.evaluate(X_test, y_test)
   print("loss: ", s[0], "accuracy: ", s[1])


TODO
----

* refresh memory on how to implement neural networks (maybe
  visit some tutorials of the machine learning practical
  course)

* learn how to program spinnaker

* find a nice linear algebra package for spinnaker
  (CBLAS/LAPACKE or some derivative maybe?)


Literature
----------

* `TensorFlow:Large-Scale Machine Learning on Heterogeneous Distributed Systems <http://download.tensorflow.org/paper/whitepaper2015.pdf>`_


Links
-----

* `spinnaker wiki <http://spinnakermanchester.github.io/>`_

* `keras <https://keras.io>`_


.. _spinnaker: http://apt.cs.manchester.ac.uk/projects/SpiNNaker/
.. _TPUs: https://en.wikipedia.org/wiki/Tensor_processing_unit
.. _GPGPUs: https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units
.. _mnist: http://yann.lecun.com/exdb/mnist/
