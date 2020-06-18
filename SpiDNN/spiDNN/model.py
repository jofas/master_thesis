import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utilities.globals_variables import \
    get_simulator

from spinn_front_end_common.utilities.connections import \
    LiveEventConnection

from spinn_utilities.socket_address import SocketAddress


from spiDNN.util import absolute_path_from_home, ReceivingLiveOutputProgress, \
    uint32t_to_float, float_to_uint32t

import spiDNN.globals as globals

from spiDNN.layers import Input, Extractor


import time


import numpy as np


class Model:
    def __init__(self):
        self.__weights = []
        self._layers = []

    def add(self, layer, layer_name=None):
        # TODO: here control correct usage

        # TODO: make sure layer_names are unique

        if layer_name is None:
            name = type(layer).__name__
            layer.name = "{}{}".format(len(self._layers), name)
        else:
            layer.name = layer_name

        if len(self._layers) > 0:
            source_layer = self._layers[-1]
            self.__weights += layer.generate_weights(source_layer)

        self._layers.append(layer)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        result = np.empty(
            (X.shape[0], self._layers[-1].atoms), dtype=np.float32)

        extractor = Extractor("extract_predictions")

        self._setup_front_end(1)

        extractor.init_neurons()
        self._init_neurons()
        self._connect_layers_forward()

        # connect extractor to the output layer
        extractor.connect_incoming(self._layers[-1])

        conn = self._setup_live_event_connection(extractor, X, result)

        front_end.run()
        front_end.stop()
        conn.close()

        return result

    def fit(self, X, y, loss_fn, epochs, batch_size):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        K = self._layers[-1].atoms

        assert y.shape[1] == K

        # start with loss_fn = "mean_squared_error"
        # mean -> 1 / K

        # backward_pass graph
        #
        # loss_unit C:
        #              counter for both -> both full -> compute loss
        #              send loss backwards
        #
        # trainable: do forward receive than wait for backward pass
        #            compute gradient descent
        #            if counter == batch_size: update weight
        #            pass E_i backwards to prev layer
        #            E_i -> know size of next layer (1 for out_layer)
        #
        # optimizer interface (in optimizations or after thesis)

        # currently no optimizer interface... just put stuff
        # (learning rate) into trainable neurons

        loss_layer = Loss("loss_unit", loss_fn, K)
        y_injectors = Input(K)
        pong = Extractor()

        self._setup_front_end(y_injectors.atoms + 2)

        loss_layer.init_neurons()
        y_injectors.init_neurons(1)
        pong.init_neurons()

        # TODO: make sure partitiony is consecutive (should be though,
        #       because y_injectors are only in this one partition)
        #       implement a test in loss_machine_vertex
        loss_layer.connect_incoming(y_injectors, partition="PARTITIONY")


        self._init_neurons()
        self._connect_layers_forward(loss_layer)
        # motherfucker I need a backward partition
        # in trainable_perceptrons a constraint... not gonna work
        #                                          with softmax
        # fuck fuck fuck
        #
        # I know how many partitions: global, softmax, backward, y
        #
        #
        # once in fit or predict, graph becomes immutable
        # that means I know how many neurons and what type of neurons
        # that means I can cluster the key space
        # that means I need to build a class with generators that
        # handles all the tricky business with dem keys and then
        # all vertices I spawn have a key constraint
        # which means I need to build wrapper around injectors and
        # extractors
        #
        # only question do I hand the key thingy down to vertices?
        # hell of a way to travel from model down to vertex (well,
        # through the layers... I think it is clearer if I hand them
        # down)

        # TODO: backward conn
        #       connect each layer beginning at loss_layer
        #       and then connect first hidden layer with pong
        #       (each neuron one connection to pong)

        # init_neurons (trainable)
        # forward pass graph
        # backward pass graph

        # live event conn doing ping pong with the board

        front_end.run(1)

        self._extract_weights()

        front_end.stop()

        # conn.close()

    def _setup_front_end(self, additional_units_count):
        n_cores = self._all_atoms() + additional_units_count

        front_end.setup(
            n_chips_required=n_cores // globals.cores_per_chip,
            model_binary_folder=absolute_path_from_home(),
            machine_time_step=globals.machine_time_step,
            time_scale_factor=globals.time_scale_factor,
        )

        self._add_db_sock()

        available_cores = \
            front_end.get_number_of_available_cores_on_machine()

        if available_cores <= n_cores:
            raise KeyError(
                "SpiNNaker doesn't have enough cores to run Model")

    def _init_neurons(self):
        """
        Initializes all Neurons (MachineVertices) in self._layers.
        """
        # Input unit needs to know how many neurons it is connected
        # to
        #
        # TODO: how will this look with Conv2D????
        #
        self._layers[0].init_neurons(self._layers[1].atoms)

        i = 0
        for layer in self._layers[1:]:
            layer.init_neurons(self.__weights[i], self.__weights[i+1])
            i += 2

    def _connect_layers_forward(self):
        """
        Builds the forward connection between each layer.
        """
        for i, layer in enumerate(self._layers[1:]):
            source_layer = self._layers[i]
            layer.connect_incoming(source_layer)

    def _connect_layers_backward(self):
        # meh
        # need to connect self._layers[1] with lpg (pong)
        # every following layer must be connected to the previous
        # (source_layer = self, prev_layer =
        pass

    def _extract_weights(self):
        i = 0
        for layer in self._layers[1:]:
            self.__weights[i:i+2] = layer.extract_weights()
            i += 2

    def _setup_live_event_connection(self, extractor, X, result):
        send_labels = self._layers[0].labels
        receive_labels = self._layers[-1].labels

        conn = LiveEventConnection(
            extractor.labels[0],
            receive_labels=receive_labels,
            send_labels=send_labels,
            machine_vertices=True
        )

        injector_callback = self._generate_injector_callback(send_labels, X)
        extractor_callback = self._generate_extractor_callback(
            receive_labels, result)

        for label in receive_labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in send_labels:
            conn.add_start_resume_callback(label, injector_callback)

        return conn

    def _generate_injector_callback(self, send_labels, X):
        send_label_to_pos = \
            {label: i for i, label in enumerate(send_labels)}

        def injector_callback(label, conn):
            for i, x in enumerate(X):
                """
                print("sending {} at step {} to neuron: {}".format(
                    x[send_label_to_pos[label]], i, label
                ))
                """
                conn.send_event_with_payload(
                    label, 0, float_to_uint32t(x[send_label_to_pos[label]])
                )

                time.sleep(0.075)

        return injector_callback

    def _generate_extractor_callback(self, receive_labels, result):
        rlop = ReceivingLiveOutputProgress(result.shape[0], receive_labels)

        def extractor_callback(label, _, val):
            val = uint32t_to_float(val)
            # print("received val: {}, neuron: {}".format(val, label))
            x = rlop.received(label)
            y = rlop.label_to_pos(label)

            result[x, y] = val

            if rlop.simulation_finished:
                front_end.stop_run()

        return extractor_callback

    def _add_db_sock(self):
        database_socket = SocketAddress(
            listen_port=globals.ack_port,
            notify_host_name=globals.host,
            notify_port_no=globals.notify_port
        )

        get_simulator().add_socket_address(database_socket)

    def _all_atoms(self):
        return sum([layer.atoms for layer in self._layers])

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        # TODO: make sure no bullshit is going on
        self.__weights = weights

    def set_weights_from_keras(self, weights):
        self.set_weights([w.read_value().numpy() for w in weights])
