# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from enum import Enum
import struct
import math
from spinn_utilities.overrides import overrides
from pacman.executor.injection_decorator import inject_items
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, VariableSDRAM
from pacman.utilities.utility_calls import is_single
from spinn_front_end_common.utilities.constants import (
    SYSTEM_BYTES_REQUIREMENT, BYTES_PER_WORD)
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)
from spinn_front_end_common.abstract_models.impl import (
    MachineDataSpecableVertex)
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)
from spinnaker_graph_front_end.utilities import SimulatorVertex
from spinnaker_graph_front_end.utilities.data_utils import (
    generate_system_data_region)


class ConwayBasicCell(SimulatorVertex, MachineDataSpecableVertex):
    """ Cell which represents a cell within the 2d fabric
    """

    PARTITION_ID = "NEIGHBOR_CONNECT"

    _MAX_OFFSET_DENOMINATOR = 10
    _INSTANCE_COUNTER = 0
    _ALL_VERTICES = 0

    PARAMS_DATA_SIZE = 4 * BYTES_PER_WORD  # has key and key

    # Regions for populations
    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[('SYSTEM', 0),
               ('PARAMS', 1),
               ])

    def __init__(self, label, state, constraints=[]):
        super(ConwayBasicCell, self).__init__(
            label,
            "conways_cell.aplx",
            constraints=constraints
        )

        ConwayBasicCell._ALL_VERTICES += 1

        # app specific data items
        self._state = bool(state)

    @inject_items({"data_n_time_steps": "DataNTimeSteps"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments={"data_n_time_steps"})
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            data_n_time_steps):

        # Generate the system data region for simulation .c requirements
        generate_system_data_region(spec, self.DATA_REGIONS.SYSTEM.value,
                                    self, machine_time_step, time_scale_factor)

        # reserve memory regions
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.PARAMS.value,
            size=self.PARAMS_DATA_SIZE, label="params")

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)
        if not is_single(partitions):
            raise ConfigurationException(
                "Can only handle one type of partition.")


        # check for duplicates
        edges = list(machine_graph.get_edges_ending_at_vertex(self))
        if len(edges) != 8:
            raise ConfigurationException(
                "I've not got the right number of connections. I have {} "
                "instead of 8".format(
                    len(machine_graph.get_edges_ending_at_vertex(self))))

        for edge in edges:
            if edge.pre_vertex == self:
                raise ConfigurationException(
                    "I'm connected to myself, this is deemed an error"
                    " please fix.")

        # write key needed to transmit with
        key = routing_info.get_first_key_from_pre_vertex(
            self, self.PARTITION_ID)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.PARAMS.value)
        spec.write_value(0 if key is None else 1)
        spec.write_value(0 if key is None else key)

        # compute offset for setting phase of conways cell
        max_offset =  machine_time_step * time_scale_factor \
                   // ConwayBasicCell._MAX_OFFSET_DENOMINATOR

        offset = int(
              math.ceil(max_offset / ConwayBasicCell._ALL_VERTICES)
            * ConwayBasicCell._INSTANCE_COUNTER
        )

        spec.write_value(offset)

        ConwayBasicCell._INSTANCE_COUNTER += 1

        """
        print("{}: ALL: {}, COUNTER: {}".format(
            self.label,
            ConwayBasicCell._ALL_VERTICES,
            ConwayBasicCell._INSTANCE_COUNTER
        ))
        """

        # write state value
        spec.write_value(int(bool(self._state)))

        # End-of-Spec:
        spec.end_specification()

    def get_data(self, buffer_manager, placement):
        # for buffering output info is taken form the buffer manager
        # get raw data, convert to list of booleans
        raw_data, data_missing = buffer_manager.get_data_by_placement(
            placement, 0)

        # do check for missing data
        if data_missing:
            print("missing_data from ({}, {}, {}); ".format(
                placement.x, placement.y, placement.p))

        # return the data, converted to list of booleans
        return [
            bool(element)
            for element in struct.unpack(
                "<{}I".format(len(raw_data) // BYTES_PER_WORD), raw_data)]

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT + self.PARAMS_DATA_SIZE)
        per_timestep_sdram = 0
        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return self.label
