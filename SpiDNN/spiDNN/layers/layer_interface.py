from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractproperty, \
    abstractmethod


@add_metaclass(AbstractBase)
class LayerInterface(object):
    """
    Interface with all the properties and methods a layer has to
    provide.
    """

    @abstractproperty
    def n_filters(self):
        """
        Number of filters the neurons of this layer instance provide.
        Default is one but convolutional layers may have more then
        one filter.
        """

    @abstractproperty
    def flatten(self):
        """
        Bool whether layer needs to be flatten (e.g. connecting con-
        volutional layers to a dense layer.
        """

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
    def connect_incoming_unique(self, source_layer, base_name):
        """
        Connects the neurons of this layer with the neurons in
        source_layer, such that the generated connection is a directed
        edge towards the neurons of this layer. Each connection has
        its own partition.

        The unique partitions are touched in such a way that the
        destination neurons see the source neurons consecutively.
        """

    @abstractmethod
    def reset(self):
        """
        Resets layer once the simulation has finished (all neurons are
        deleted).
        """
