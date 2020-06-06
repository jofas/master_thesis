from spinn_front_end_common.utilities.constants import NOTIFY_PORT

machine_time_step = 25
time_scale_factor = 1
cores_per_chip = 15
partition_name = "PARTITION0"
ack_port = 22222
host = "127.0.0.1"
notify_port = NOTIFY_PORT

activations = {fn_name: i for i, fn_name in enumerate([
    "identity", "relu", "sigmoid", "tanh", "softmax"
])}
