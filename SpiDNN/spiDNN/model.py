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
        self._reset_layers()

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

        # trainable: do forward receive than wait for backward pass
        #            compute gradient descent
        #            if counter == batch_size: update weight
        #            pass E_i backwards to prev layer

        loss_layer = Loss("loss_unit", loss_fn, K)
        y_injectors = Input(K, label="YInjector")
        pong = Extractor("pong")

        n_cores = self._all_neurons() + y_injectors.n_neurons + 2
        gfe.setup(n_cores)

        loss_layer.init_neurons()
        y_injectors.init_neurons(neurons_next_layer=1)
        pong.init_neurons()
        self._init_neurons(
            trainable=True, batch_size=batch_size, learning_rate=learning_rate)

        self._connect_layers_forward()

        loss_layer.connect_incoming(
            self._layers[-1], globals.forward_partition)

        loss_layer.connect_incoming(y_injectors, globals.y_partition)

        self._connect_layers_backward()

        self._layers[-1].connect_incoming_unique(
            loss_layer, base_name=globals.backward_partition)

        pong.connect_incoming(
            self._layers[1], globals.backward_partition)

        conn = self._setup_fit_live_event_connection(
            pong, y_injectors, X, y, epochs)

        gfe.run()

        self._extract_weights()

        gfe.stop()

        conn.close()
        self._reset_layers()

    def _reset_layers(self):
        for layer in self._layers:
            layer.reset()

    def _init_neurons(
            self, trainable=False, batch_size=None, learning_rate=None):
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
                trainable=trainable,
                batch_size=batch_size,
                learning_rate=learning_rate)
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

        injector_callback = self._generate_predict_injector_callback(
            send_labels, X)
        extractor_callback = self._generate_predict_extractor_callback(
            receive_labels, result)

        for label in receive_labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in send_labels:
            conn.add_start_resume_callback(label, injector_callback)

        return conn

    def _generate_predict_injector_callback(self, send_labels, X):
        send_label_to_pos = {
            label: i for i, label in enumerate(send_labels)}

        def injector_callback(label, conn):
            for x in X:
                conn.send_event_with_payload(
                    label,
                    0,
                    util.float_to_uint32t(x[send_label_to_pos[label]]))

                time.sleep(0.075)

        return injector_callback

    def _generate_predict_extractor_callback(self, receive_labels, result):
        rlop = util.ReceivingLiveOutputProgress(
            result.shape[0], receive_labels)

        def extractor_callback(label, _, val):
            val = util.uint32t_to_float(val)
            x = rlop.received(label)
            y = rlop.label_to_pos(label)

            result[x, y] = val

            if rlop.simulation_finished:
                gfe.stop_run()

        return extractor_callback

    def _setup_fit_live_event_connection(
            self, extractor, y_injectors, X, y, epochs):
        send_labels = self._layers[0].labels + y_injectors.labels
        receive_labels = self._layers[1].labels

        conn = LiveEventConnection(
            extractor.labels[0],
            receive_labels=receive_labels,
            send_labels=send_labels,
            machine_vertices=True
        )

        barrier = Condition()

        extractor_callback = self._generate_fit_extractor_callback(
            receive_labels, X, barrier, epochs)

        y_injector_callback = self._generate_fit_injector_callback(
            y_injectors.labels, y, barrier, epochs)

        X_injector_callback = self._generate_fit_injector_callback(
            self._layers[0].labels, X, barrier, epochs)

        for label in receive_labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in y_injectors.labels:
            conn.add_start_resume_callback(label, y_injector_callback)

        for label in self._layers[0].labels:
            conn.add_start_resume_callback(label, X_injector_callback)

        return conn

    def _generate_fit_extractor_callback(
            self, receive_labels, X, barrier, epochs):
        frlop = util.FitReceivingLiveOutputProgress(
            epochs, len(X), barrier, len(receive_labels))

        def extractor_callback(label, _0, _1):
            frlop.receive()

            if frlop.received_all:
                if frlop.simulation_finished:
                    gfe.stop_run()
                else:
                    frlop.notify_injectors()
                frlop.reset()

        return extractor_callback

    def _generate_fit_injector_callback(self, send_labels, M, barrier, epochs):
        send_label_to_pos = {
            label: i for i, label in enumerate(send_labels)}

        def injector_callback(label, conn):
            barrier.acquire()
            for epoch in range(0, epochs):
                for m in M:
                    #print("sending value: {} to label: {}".format(
                    #    m[send_label_to_pos[label]], label))
                    conn.send_event_with_payload(
                        label,
                        0,
                        util.float_to_uint32t(m[send_label_to_pos[label]]))
                    barrier.wait()
                print("epoch done: {}".format(epoch))
            barrier.release()

        return injector_callback

    def _all_neurons(self):
        return sum([layer.n_neurons for layer in self._layers])

    def get_weights(self):
        return self.__weights

    def set_weights(self, weights):
        # TODO: make sure no bullshit is going on
        self.__weights = weights

    def set_weights_from_keras(self, weights):
        self.set_weights([w.read_value().numpy() for w in weights])
