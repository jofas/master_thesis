from six import add_metaclass

from spinn_utilities.abstract_base import AbstractBase, abstractproperty, \
    abstractmethod


@add_metaclass(AbstractBase)
class WeightsInterface(object):

    @abstractmethod
    def generate_weights(self, source_layer):
        """
        Generate weights for connections from source_layer to self.
        The weights are returned and re-injected into the layer
        (during neuron initialization method init_neurons).
        """

    @abstractmethod
    def extract_weights(self):
        """
        Extract weights from the board.
        """
