Deep Learning on SpiNNaker (formerly: A Tensorflow Backend to SpiNNaker)
========================================================================


See the `report <report/report.pdf>`_ for an introduction the this project.


Work Plan
---------

+---------------+----+----+----+----+
|               | 31 | 32 | 33 | 34 |
+===============+====+====+====+====+
| **IMPLEMENT** |                   |
+---------------+----+----+----+----+
| Conv          | xx |              |
+---------------+----+----+----+----+
| Pool          | xx |              |
+---------------+----+----+----+----+
| RL            | xx |              |
+---------------+----+----+----+----+
| Validation    |    | xx |         |
+---------------+----+----+----+----+
| Splitting     |    | xx |         |
+---------------+----+----+----+----+
| RIPTMCS       |    | xx |         |
+---------------+----+----+----+----+
| Cleanup Code  |    | xx |         |
+---------------+----+----+----+----+
| Document Code |    | xx |         |
+---------------+----+----+----+----+
| **BENCHMARK** |    | xx |         |
+---------------+----+----+----+----+
| **WRITING**   |                   |
+---------------+----+----+----+----+
| Hardware      | xx |              |
+---------------+----+----+----+----+
| ResNet        | xx |              |
+---------------+----+----+----+----+
| MLPerf        | xx |              |
+---------------+----+----+----+----+
| Background    | xx |              |
+---------------+----+----+----+----+
| SpiDNN        |    | xx | xx |    |
+---------------+----+----+----+----+
| Log           | xx | xx |         |
+---------------+----+----+----+----+
| Benchmark     |    | xx | xx |    |
+---------------+----+----+----+----+
| Conclusion    |         | xx |    |
+---------------+----+----+----+----+
| Proof reading |              | xx |
+---------------+----+----+----+----+


Optimizations and features
--------------------------

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
