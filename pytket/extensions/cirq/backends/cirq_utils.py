# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast

from cirq.circuits import Circuit as CirqCircuit
from cirq.devices import GridQubit, LineQubit
from cirq.ops import MeasurementGate, NamedQubit, QubitOrder
from cirq.protocols import is_measurement
from pytket.circuit import Bit, Circuit, Qubit


def _get_default_uids(
    cirq_circuit: CirqCircuit, tket_circuit: Circuit
) -> tuple[list[Bit], list[Qubit]]:
    if len(tket_circuit.qubit_readout) == 0:
        return [], tket_circuit.qubits
    ordered_cirq_qubits = QubitOrder.as_qubit_order(QubitOrder.DEFAULT).order_for(
        cirq_circuit.all_qubits()
    )

    ordered_tket_qubits = []
    for cirq_qubit in ordered_cirq_qubits:
        if isinstance(cirq_qubit, NamedQubit):
            ordered_tket_qubits.extend(
                [qb for qb in tket_circuit.qubits if qb.reg_name == cirq_qubit.name]
            )
        if isinstance(cirq_qubit, LineQubit):
            ordered_tket_qubits.extend(
                [qb for qb in tket_circuit.qubits if qb.index == cirq_qubit.x]
            )
        if isinstance(cirq_qubit, GridQubit):
            ordered_tket_qubits.extend(
                [
                    qb
                    for qb in tket_circuit.qubits
                    if (qb.index[0] == cirq_qubit.row and qb.index[1] == cirq_qubit.col)
                ]
            )

    cirq_measures = [c[1] for c in cirq_circuit.findall_operations(is_measurement)]
    tket_bit_to_qubit_map = {b: q for q, b in tket_circuit.qubit_to_bit_map.items()}
    ordered_tket_bits = []
    for cirq_qubit in ordered_cirq_qubits:
        for cirq_measure in cirq_measures:
            if len(cirq_measure.qubits) > 1:
                raise ValueError(
                    "Cirq Qubit measurement assigned to multiple classical bits."
                )
            if cirq_measure.qubits[0] == cirq_qubit:
                for tket_bit, _ in tket_bit_to_qubit_map.items():  # noqa: PERF102
                    if cast("MeasurementGate", cirq_measure.gate).key == str(tket_bit):
                        ordered_tket_bits.append(tket_bit)
    return (ordered_tket_bits, ordered_tket_qubits)
