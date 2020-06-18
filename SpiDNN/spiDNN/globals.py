from spinn_front_end_common.utilities.constants import NOTIFY_PORT

machine_time_step = 500
time_scale_factor = 10
cores_per_chip = 17

forward_partition = "PARTITION_FORWARD"
backward_partition = "PARTITION_BACKWARD"
softmax_partition = "PARTITION_SOFTMAX"
y_partition = "PARTITION_Y"

ack_port = 22222
host = "127.0.0.1"
notify_port = NOTIFY_PORT

_max_offset_factor = 0.1
max_offset = int(machine_time_step * time_scale_factor * _max_offset_factor)

activations = {fn_name: i for i, fn_name in enumerate([
    "identity", "relu", "sigmoid", "tanh", "softmax"
])}

losses = {fn_name: i for i, fn_name in enumerate([
    "mean_squared_error"
])}
