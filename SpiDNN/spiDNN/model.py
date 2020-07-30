from spinn_front_end_common.utilities.connections import \
    LiveEventConnection

import spiDNN.gfe as gfe
import spiDNN.util as util
import spiDNN.globals as globals
from spiDNN.layers import Input, Extractor, Loss

import time
from threading import Condition

import numpy as np


class Model:
    def __init__(self):
        self.__weights = []
        self._layers = []

    def add(self, layer, label=None):
        # TODO: here control correct usage

        # TODO: make sure layer_names are unique
        #       ... remove labeling process???

        if label is None:
            name = type(layer).__name__
            layer.label = "{}{}".format(len(self._layers), name)
        else:
            layer.label = label

        if len(self._layers) > 0:
            source_layer = self._layers[-1]
            self.__weights += layer.generate_weights(source_layer)

        self._layers.append(layer)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)

        result = np.empty(
            (X.shape[0], self._layers[-1].n_neurons), dtype=np.float32)

        extractor = Extractor("extract_predictions")

        self._reset_layers()

        gfe.setup(self._all_neurons() + 1)

        extractor.init_neurons()
        self._init_neurons()

        self._connect_layers_forward()
        # connect extractor to the output layer
        extractor.connect_incoming(self._layers[-1], globals.forward_partition)

        conn = self._setup_predict_live_event_connection(extractor, X, result)

        gfe.run()
        gfe.stop()
        conn.close()
        return result

    def fit(self, X, y, loss_fn, epochs, batch_size, learning_rate):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        K = self._layers[-1].n_neurons

        assert y.shape[1] == K
        assert X.shape[1] == self._layers[0].n_neurons

        assert len(X) == len(y)

        if batch_size > len(X):
            batch_size = len(X)

        trainable_params = util.TrainableParams(
            epochs, len(X), batch_size, learning_rate)

        loss_layer = Loss("loss_unit", loss_fn, K)
        y_injectors = Input(K, label="YInjector")
        pong = Extractor("pong")

        self._reset_layers()

        n_cores = self._all_neurons() + y_injectors.n_neurons + 2
        gfe.setup(n_cores)

        # init neurons
        loss_layer.init_neurons(trainable_params=trainable_params)
        y_injectors.init_neurons(neurons_next_layer=1)
        pong.init_neurons()
        self._init_neurons(trainable_params=trainable_params)

        # forward pass
        self._connect_layers_forward()
        loss_layer.connect_incoming(
            self._layers[-1], globals.forward_partition)

        # connections for streaming true labels onto board for computing
        # the loss
        loss_layer.connect_incoming(y_injectors, globals.y_partition)

        # backward pass
        self._connect_layers_backward()
        self._layers[-1].connect_incoming_unique(
            loss_layer, base_name=globals.backward_partition)
        pong.connect_incoming(
            self._layers[1], globals.backward_partition)
        pong.connect_incoming(loss_layer, globals.backward_partition)

        conn = self._setup_fit_live_event_connection(
            pong, loss_layer, y_injectors, X, y, epochs)

        gfe.run()

        # race condition when writing weights back to sdram
        time.sleep(2)
        self._extract_weights()

        gfe.stop()
        conn.close()

    def _reset_layers(self):
        for layer in self._layers:
            layer.reset()

    def _init_neurons(self, trainable_params=None):
        """
        Initializes all Neurons (MachineVertices) in self._layers.
        """
        # Input unit needs to know how many neurons it is connected
        # to
        #
        # TODO: how will this look with Conv2D????
        #
        self._layers[0].init_neurons(
            neurons_next_layer=self._layers[1].n_neurons)

        i = 0
        for layer in self._layers[1:]:
            # TODO: how will this look wit more than just Dense
            #       layers?
            layer.init_neurons(
                weights=self.__weights[i],
                biases=self.__weights[i+1],
                trainable_params=trainable_params)
            i += 2

    def _connect_layers_forward(self):
        """
        Builds the forward connection between each layer in self._layer.
        """
        for i, layer in enumerate(self._layers[1:]):
            source_layer = self._layers[i]
            layer.connect_incoming(source_layer, globals.forward_partition)

    def _connect_layers_backward(self):
        """
        Builds the backward connection between each layer in self._layer
        (except Input layer).
        """
        i = 2
        for layer in self._layers[1:-1]:
            source_layer = self._layers[i]
            layer.connect_incoming(
                source_layer, globals.backward_partition)
            i += 1

    def _extract_weights(self):
        i = 0
        for layer in self._layers[1:]:
            self.__weights[i:i+2] = layer.extract_weights()
            i += 2

    def _setup_predict_live_event_connection(self, extractor, X, result):
        send_labels = self._layers[0].labels
        receive_labels = self._layers[-1].labels

        conn = LiveEventConnection(
            extractor.labels[0],
            receive_labels=receive_labels,
            send_labels=send_labels,
            machine_vertices=True
        )

        barrier = Condition()

        extractor_callback = self._generate_predict_extractor_callback(
            receive_labels, result, barrier)

        injector_callback = self._generate_predict_injector_callback(
            send_labels, X, barrier)

        for label in receive_labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in send_labels:
            conn.add_start_resume_callback(label, injector_callback)

        return conn

    def _generate_predict_extractor_callback(
            self, receive_labels, result, barrier):
        extractor_manager = util.PingPongExtractionManager(
            1, result.shape[0], len(receive_labels))

        label_to_pos = {
            label: i for i, label in enumerate(receive_labels)}

        def extractor_callback(label, _, val):
            x = extractor_manager.receive()

            val = util.uint32t_to_float(val)
            result[x, label_to_pos[label]] = val

            if extractor_manager.received_all:
                if extractor_manager.simulation_finished:
                    gfe.stop_run()
                else:
                    with barrier:
                        barrier.notify_all()
                extractor_manager.reset()

        return extractor_callback

    def _generate_predict_injector_callback(self, send_labels, X, barrier):
        send_label_to_pos = {
            label: i for i, label in enumerate(send_labels)}

        # TODO: how will this look with 2D input?
        if len(X.shape) == 2:
            X = X.reshape(*X.shape, 1)

        def injector_callback(label, conn):
            barrier.acquire()
            for x in X:
                conn.send_events_with_payloads(
                    label,
                    [(0, util.float_to_uint32t(
                        x[send_label_to_pos[label],i]))
                        for i in range(0, X.shape[-1])])
                barrier.wait()
            barrier.release()

        return injector_callback

    def _setup_fit_live_event_connection(
            self, extractor, loss_layer, y_injectors, X, y, epochs):

        conn = LiveEventConnection(
            extractor.labels[0],
            receive_labels=self._layers[1].labels + loss_layer.labels,
            send_labels=self._layers[0].labels + y_injectors.labels,
            machine_vertices=True
        )

        barrier = Condition()

        loss_extraction_manager = util.LossExtractionManager(
            epochs, len(X))

        def extractor_loss_callback(label, _, loss):
            loss_extraction_manager.receive(loss)

        extractor_callback = self._generate_fit_extractor_callback(
            self._layers[1].labels, X, barrier, epochs)

        y_injector_callback = self._generate_fit_injector_callback(
            y_injectors.labels, y, barrier, epochs)

        X_injector_callback = self._generate_fit_injector_callback(
            self._layers[0].labels, X, barrier, epochs)

        for label in loss_layer.labels:
            conn.add_receive_callback(label, extractor_loss_callback)

        for label in self._layers[1].labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in y_injectors.labels:
            conn.add_start_resume_callback(label, y_injector_callback)

        for label in self._layers[0].labels:
            conn.add_start_resume_callback(label, X_injector_callback)

        return conn

    def _generate_fit_extractor_callback(
            self, receive_labels, X, barrier, epochs):
        extractor_manager = util.PingPongExtractionManager(
            epochs, len(X), len(receive_labels) * X.shape[1])

        def extractor_callback(label, _0, _1):
            extractor_manager.receive()

            if extractor_manager.received_all:
                if extractor_manager.simulation_finished:
                    gfe.stop_run()
                else:
                    with barrier:
                        barrier.notify_all()
                extractor_manager.reset()

        return extractor_callback

    def _generate_fit_injector_callback(self, send_labels, M, barrier, epochs):
        send_label_to_pos = {
            label: i for i, label in enumerate(send_labels)}

        # TODO: how will this look with 2D input?
        if len(M.shape) == 2:
            M = M.reshape(*M.shape, 1)

        def injector_callback(label, conn):
            barrier.acquire()
            for epoch in range(0, epochs):
                for m in M:
                    conn.send_events_with_payloads(
                        label,
                        [(0, util.float_to_uint32t(
                            m[send_label_to_pos[label],i]))
                            for i in range(0, M.shape[-1])])
                    barrier.wait()
            barrier.release()

        return injector_callback

    def _all_neurons(self):
        return sum([layer.n_neurons for layer in self._layers])

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        # TODO: make sure no bullshit is going on
        for i, weight in enumerate(weights):
            assert self.__weights[i].shape == weight.shape

        self.__weights = weights
