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

To answer this question, a deeper look at what tensorflow
really is shows, that tensorflow is not only the wrong
level of abstraction for interfacing with the spinnaker
runtime, but it is also incompatible (I think) or at least
unfeasible, since tensorflow in itself is its own runtime.

TODO: describe tensorflow from the whitepaper
What is tensorflow? While it is common to simply say
tensorflow is a machine learning library, in fact, it is
not. According to the whitepaper [CITE], tensorflow is
simply notation

machine learning -- can be broken down to glorified linear
algebra

describe why incompatible

tf is its own complex runtime - incompatible

tf wrong target

tf vast ecosystem

example: keras tf graph before - after training - conceptual graph

conceptual graph - more freedom for own implementation


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

   # convert to spinnaker graph
   model = keras2spinnaker(model)

   # runs spinnaker graph and update the keras model with
   # the weights
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
