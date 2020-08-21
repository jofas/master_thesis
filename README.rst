Deep Learning on SpiNNaker
==========================

See the `report <report/report.pdf>`_ for an introduction the this project.
The master thesis discussing SpiDNN in detail can be found
`here <thesis/thesis.pdf>`_.


Installation
------------

First, in order to compile the binaries for the SpiNNaker toolchain
and for SpiDNN, you need to download the `GNU ARM Embedded Toolchain <https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm>`_
and make it available in your path.

Afterwards, download and install the SpiNNaker toolchain with the
graph front-end:

.. code-block:: bash

   git clone https://github.com/SpiNNakerManchester/SupportScripts
   bash SupportScripts/install.sh gfe
   bash SupportScritps/setup.sh
   bash SupportScripts/automatic_make.sh

Other dependecies that must be installed for SpiDNN:

* numpy

  ``pip install numpy``

* tensorflow

  ``pip install tensorflow``

* keras

  ``pip install keras``

Afterwards, enter the ``SpiDNN/`` directory and execute ``source scripts/env.sh``.
This adds SpiDNN to your python-path variable.
Also from the ``SpiDNN/`` directory, execute ``make``, which will build
the binaries for SpiNNaker.
In the ``SpiDNN/tests/test_inference.py`` and ``SpiDNN/tests/test_training.py``
files, you will find examples showing how SpiDNN is used.


TODO: Optimizations and features
--------------------------------

* custom injectors which support partitions

* pong receive without payload (less network traffic, more complexity
  in neurons)

* split neurons (bigger layers)

* quantization (speed and memory size)

* populations or some other mechanism to combine neurons (saving cores)

* optimizer interface

* loss into perceptron (saving cores but higher complexity)

* progress interface with epoch loss (loss computed by host?)

* validation interface and more metrics, which could be a nice way for
  early stopping (then definetly loss computed
  on host (error still on the board) (see previous bullet point))

* monitor/sync core for each layer (simplifying interface)

* enable multiple successive sessions (currently LiveEventConnection
  does some weird stuff)

* export/import weights to/from file (fancy HDF5 or json which is less
  sexy)

* nicer injection of weights into machine_vertices (different interface
  than just using init_neurons)


Links
-----

* `spinnaker project <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/project/>`_

* `spinnaker wiki <http://spinnakermanchester.github.io/>`_

* `MLPerf <https://mlperf.org/>`_

* `Identity Mappings in Deep Residual Networks <https://arxiv.org/abs/1603.05027>`_

* `ResNet-50v1.5 <https://github.com/facebookarchive/fb.resnet.torch>`_
