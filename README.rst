Deep Learning on SpiNNaker (formerly: A Tensorflow Backend to SpiNNaker)
========================================================================


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

1. Train the following neural networks [1]_ on spinnaker
   and compare the raw speed of training to state of the
   art approaches for training (to be determined which):

   * 1-hidden-layer fully connected NN

   * 2-hidden-layer fully connected NN

   * LeNet-1

   * LeNet-4

   * LeNet-5

   * Boosted LeNet-4

2. While the first benchmark would give information on the
   performance of spinnaker, I personally believe,
   comparing a technology that can scale to the size of a
   supercomputer against unscalable accelerated
   architectures could be misleading.
   Therfore, I would like to use the same benchmark to
   analyze the energy efficiency of spinnaker as well, to
   see whether spinnaker can be a more scalable and more
   cost effective in terms of energy consumption.

.. [1] For more information see `here <https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17>`_


Why I propose a different benchmark, then the one described in the project proposal
-----------------------------------------------------------------------------------

Alan's project proposal wants me to implement the backend
and benchmark it against the SSNToolbox_, developed by
researchers at the ETH Zürich.

If one reads the introduction to the SSNToolbox, it
explicitly states:

   *Training a deep spiking network (i.e. learning the
   synaptic weights) is difficult.
   An alternative approach is to take a pre-trained neural
   network and convert it into a spiking neural network.*

The SSNToolbox converts pretrained NNs to spiking NNs,
which is the original target of spinnaker.
This is not at all, what we want to achieve.
We want to find out, if spinnaker is a competitive target
for training deep NNs, not doing inference with it.


Why Deep Learning on SpiNNaker instead of A Tensorflow Backend to SpiNNaker
---------------------------------------------------------------------------

I'm making the daring assumption, that implementing a new
backend for tensorflow is not possible for one person in
three months, without prior knowledge of
`XLA <https://www.tensorflow.org/xla>`_ (implementing an
XLA backend is significantly simpler than retargeting the
tensorflow operations directly, according to [2]_),
`llvm <http://llvm.org>`_ and C++ (regretably, I have no
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
library `keras <https://keras.io>`_, the de facto standard
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

.. [TF2015] `TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems <http://download.tensorflow.org/paper/whitepaper2015.pdf>`_


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


   # the main (and only?) interface this dissertation
   # will implement.
   #
   # It parses the conceptual graph of the model to
   # a spinnaker graph and runs it on a spinnaker
   # machine.
   # Afterwards, the weights (trainable parameters)
   # of the keras model are updated and the model can
   # be used for inference.
   k2s.fit(model, X_train, y_train)

   # runs on local machine
   s = model.evaluate(X_test, y_test)
   print("loss: ", s[0], "accuracy: ", s[1])


TODO
----

* Refresh memory on how to implement neural networks (maybe
  visit some tutorials of the machine learning practical
  course)

* Learn how to program spinnaker


Other bits and bobs
-------------------

* I don't see myself having time to spare, but if this
  should be the case, the project can be arbitrarily scaled
  up.

  For instance, for the proposed benchmark, I'd just have
  to implement a subset of keras (indeed why I focus on a
  single task, image recognition), which can be increased
  to a workload, a single person can not implement in three
  months (with a reasonable amount of sleep in it).

  Otherwise, implementing an interface for doing inference
  on spinnaker can be done as well (then we could actually
  benchmark against SSNToolbox_).


Literature
----------

* `TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems <http://download.tensorflow.org/paper/whitepaper2015.pdf>`_

* `Deep Learning Performance on Different Architectures <https://static.epcc.ed.ac.uk/dissertations/hpc-msc/2017-2018/Spyro_Nita-dissertation-spyro-nita.pdf>`_

Links
-----

* `spinnaker project <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/project/>`_

* `spinnaker wiki <http://spinnakermanchester.github.io/>`_

* `keras <https://keras.io>`_


.. _spinnaker: http://apt.cs.manchester.ac.uk/projects/SpiNNaker/
.. _TPUs: https://en.wikipedia.org/wiki/Tensor_processing_unit
.. _GPGPUs: https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units
.. _mnist: http://yann.lecun.com/exdb/mnist/
.. _SSNToolbox: https://snntoolbox.readthedocs.io/en/latest/guide/intro.html
