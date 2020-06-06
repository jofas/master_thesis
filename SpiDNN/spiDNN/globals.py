from spinn_front_end_common.utilities.constants import NOTIFY_PORT

machine_time_step = 3000
time_scale_factor = 10
cores_per_chip = 17
partition_name = "PARTITION0"
ack_port = 22222
host = "127.0.0.1"
notify_port = NOTIFY_PORT

_max_offset_denominator = 2
max_offset = machine_time_step * time_scale_factor // _max_offset_denominator

activations = {fn_name: i for i, fn_name in enumerate([
    "identity", "relu", "sigmoid", "tanh", "softmax"
])}
