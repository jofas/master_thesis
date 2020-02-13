Deep Learning on SpiNNaker (formerly: A Tensorflow Backend to SpiNNaker)
========================================================================


See the `report <report/report.pdf>`_ for an introduction the this project.


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


Links
-----

* `spinnaker project <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/project/>`_

* `spinnaker wiki <http://spinnakermanchester.github.io/>`_
