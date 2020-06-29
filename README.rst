Deep Learning on SpiNNaker (formerly: A Tensorflow Backend to SpiNNaker)
========================================================================


See the `report <report/report.pdf>`_ for an introduction the this project.


Work Plan
---------

+---------------+----+----+----+----+----+----+----+----+----+----+
|               | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 |
+===============+====+====+====+====+====+====+====+====+====+====+
| **IMPLEMENT** |                                                 |
+---------------+----+----+----+----+----+----+----+----+----+----+
| SGD           | xx |                                            |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Conv          |    | xx | xx |                                  |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Pool          |              | xx |                             |
+---------------+----+----+----+----+----+----+----+----+----+----+
| RL            |                   | xx |                        |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Optimize      |                        | xx | xx | xx |         |
+---------------+----+----+----+----+----+----+----+----+----+----+
| **BENCHMARK** |                        | xx | xx | xx |         |
+---------------+----+----+----+----+----+----+----+----+----+----+
| **WRITING**   |                                                 |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Hardware      | xx |                                            |
+---------------+----+----+----+----+----+----+----+----+----+----+
| ResNet        |    | xx |                                       |
+---------------+----+----+----+----+----+----+----+----+----+----+
| MLPerf        |         | xx |                                  |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Deep Learning |              | xx | xx |                        |
+---------------+----+----+----+----+----+----+----+----+----+----+
| SpiDNN        |                        | xx | xx | xx |         |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Log           | xx | xx | xx | xx | xx | xx | xx | xx |         |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Benchmark     |                                  | xx | xx |    |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Conclusion    |                                       | xx |    |
+---------------+----+----+----+----+----+----+----+----+----+----+
| Proof reading |                   | xx |              | xx | xx |
+---------------+----+----+----+----+----+----+----+----+----+----+


Optimizations and features
--------------------------

* custom injectors which support partitions

* pong receive without payload

* split neurons (bigger layers)

* quantization (speed and memory size)

* populations or some other mechanism to combine neurons (saving cores)

* optimizer interface

* loss into perceptron (saving cores but higher complexity)

* progress interface with epoch loss (loss computed by host?)

* monitor/sync core for each layer (simplifying interface)

* weight extraction (dtcm -> sdram) only when training is finished,
  not after each batch (speed)


Links
-----

* `spinnaker project <http://apt.cs.manchester.ac.uk/projects/SpiNNaker/project/>`_

* `spinnaker wiki <http://spinnakermanchester.github.io/>`_

* `MLPerf <https://mlperf.org/>`_

* `Identity Mappings in Deep Residual Networks <https://arxiv.org/abs/1603.05027>`_

* `ResNet-50v1.5 <https://github.com/facebookarchive/fb.resnet.torch>`_
