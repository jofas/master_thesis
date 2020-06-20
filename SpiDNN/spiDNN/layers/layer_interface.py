from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractproperty, \
    abstractmethod


import spiDNN.globals as globals


@add_metaclass(AbstractBase)
class LayerInterface(object):

    @abstractproperty
    def n_neurons(self):
        """
        Number of machine vertices contained in layer.
        """

    @abstractproperty
    def neurons(self):
        """
        The neurons (machine vertices) of the layer.
        """

    @abstractproperty
    def label(self):
        """
        Label of the layer.
        """

    @abstractproperty
    def labels(self):
        """
        Labels of the neurons of the layer.
        """

    @abstractmethod
    def init_neurons(self, **kwargs):
        """
        Initializes the neurons in n_neurons and adds them to the
        machine_graph of the GFE.
        """

    @abstractmethod
    def connect_incoming(self, source_layer, partition):
        """
        Connects the neurons of this layer with the neurons in
        source_layer, such that the generated connection is a directed
        edge towards the neurons of this layer.
        """

    @abstractmethod
    def connect_incoming_unique(self, source_layer):
        """
        Connects the neurons of this layer with the neurons in
        source_layer, such that the generated connection is a directed
        edge towards the neurons of this layer. Each connection has
        its own partition.
        """
