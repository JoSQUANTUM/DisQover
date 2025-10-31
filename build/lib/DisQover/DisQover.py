import itertools
import math
from typing import List

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from ipywidgets import HBox, Layout, VBox, interactive
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import XGate
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


class QRW_simulation:
    """
    simulator class for 1D and 2D Quantum Random Walks

    """

    def __init__(self, dim: int):
        """
        Parameters
        ----------------------
        dim: an integer
             -> dimension of QRW (either 1 or 2)
        """

        if dim not in [1, 2]:
            raise ValueError("dim is either 1  or 2.")

        self.dim = dim

    def insert_zero(self, x: np.ndarray) -> np.ndarray:
        """
        insert zeros in between values of the given 1D distribution so that it can be modeled by QRW
        """
        y = [x[int(i / 2)] if i % 2 == 0 else 0 for i in range(2 * len(x) - 1)]
        return np.array(y)

    def conv(self, vector: np.ndarray, n: int) -> np.ndarray:
        """
        Convolve an one-dimensional input vector with a delta function δ(x+n),
        which is equivalent to shifting the vector by n units to the left.

        Parameters
        ---------------------
        vector: a numpy array
                -> the vector to be convolved

        n: an integer
           -> The amount to be shifted to the left

        Returns
        ---------------------
        shift_vector: a numpy array
                      -> the convolved (shifted) vector
        """
        if n == 0:
            shift_vector = vector
        if n > 0:
            shift_vector = np.hstack((vector[n:], np.zeros(n)))
        else:
            shift_vector = np.hstack((np.zeros(-n), vector[:n]))

        return shift_vector

    def conv2d(self, tensor: np.ndarray, n: int) -> np.ndarray:
        """
        Shift a two-dimensional input tensor according to a give direction.

        Parameters
        ---------------------
        tensor: a numpy array
                -> the tensor to be shifted

        n: an integer
           -> The index for the direction of shift (0 <= n <= 3)
           -> 0: bottom left, 1: top left, 2: bottom right, 3: top right

        Returns
        ---------------------
        shift_vector: a numpy array
                      -> the convolved (shifted) vector
        """
        if n not in list(range(4)):
            raise ValueError("n should be a integer between 0 and 3")

        bit = bin(n)[2:].zfill(2)
        shift = np.zeros((3, 3))

        if bit == "00":
            tensor = np.concatenate(
                (tensor[:, 1:], np.zeros((tensor.shape[0], 1))), axis=1
            )
            tensor = np.concatenate(
                (np.zeros((1, tensor.shape[1])), tensor[:-1, :]), axis=0
            )
        if bit == "01":
            tensor = np.concatenate(
                (tensor[:, 1:], np.zeros((tensor.shape[0], 1))), axis=1
            )
            tensor = np.concatenate(
                (tensor[1:, :], np.zeros((1, tensor.shape[1]))), axis=0
            )
        if bit == "10":
            tensor = np.concatenate(
                (np.zeros((tensor.shape[0], 1)), tensor[:, :-1]), axis=1
            )
            tensor = np.concatenate(
                (np.zeros((1, tensor.shape[1])), tensor[:-1, :]), axis=0
            )
        if bit == "11":
            tensor = np.concatenate(
                (np.zeros((tensor.shape[0], 1)), tensor[:, :-1]), axis=1
            )
            tensor = np.concatenate(
                (tensor[1:, :], np.zeros((1, tensor.shape[1]))), axis=0
            )

        return tensor

    def simulate(
        self,
        coin_state: List[np.ndarray],
        U: List[np.ndarray],
        init_amp: np.ndarray,
        steps: int,
        return_amp: bool = False,
    ) -> np.ndarray:
        """
        Given an initial amplitufde distribution and relevant quantum random walks parameters, the simulate function simulates the process of
        QRW deterministically, and returns the resulting  probability distribution or amplitude distribution if specified.

        Parameters
        ----------------------------
        coin_state: a list of numpy array
                   -> the quantum states of the coins as complex vectors.

        U: a list of numpy array
           -> the unitary matrices that represent the coin flip operations.

        init_amp: a numpy array
                  -> the initial amplitude distribution of the walker.

        steps: an integer
              -> the number of steps in the QRW.

        return_amp: a boolean
                    -> specify if the output should be the amplitude distribution.

        Returns
        ----------------------------
        if return_amp=False ->
        prob: a numpy array
              -> the resulting probability distribution

        if return_amp=True ->
        tensor: a numpy array
                -> the resulting amplitude distribution
        """
        n = self.dim

        if n == 2:
            coin_amp = list(itertools.product(coin_state[0], coin_state[1]))
            coin_vec = np.array([np.prod(list(coin)) for coin in coin_amp])

        else:
            coin_vec = np.array(coin_state[0])

        tensor = np.tensordot(
            np.array(coin_vec).reshape(len(coin_vec), 1),
            init_amp.reshape((1,) + init_amp.shape),
            axes=1,
        )

        Un = U[0]
        for i in range(1, n):
            Un = np.kron(Un, U[i])

        for step in range(steps):
            if n == 2:
                tensor = np.tensordot(Un, tensor, axes=1)
                shift_tensors = [
                    self.conv2d(tensor[i], i).reshape((1,) + tensor.shape[1:])
                    for i in range(tensor.shape[0])
                ]
                tensor = np.concatenate(shift_tensors, axis=0)
            else:
                tensor = Un @ tensor
                tensor = np.vstack(
                    (self.conv(tensor[0, :], 1), self.conv(tensor[1, :], -1))
                )

        if return_amp == True:
            return tensor
        else:
            prob = np.tensordot(
                np.ones((1, tensor.shape[0])),
                (tensor * np.conjugate(tensor)).real,
                axes=1,
            ).reshape(init_amp.shape)
            return prob


class CRW_simulation:
    """
    simulator class for 1D Classical Random Walks

    """

    def __init__(self, coin_probs: np.ndarray):
        """
        Parameters
        ----------------------
        coin_probs: a numpy array or list
                    -> the probabilities of left and right movements.
        """

        self.coin_probs = coin_probs
        self.qrw = QRW_simulation(1)

    def simulate(
        self, init_prob: np.ndarray, steps: int, plot: bool = False
    ) -> np.ndarray:
        """
        Given an initial probability distribution and the number of steps, the simulate function simulates the process of
        a classical random walk, and returns the resulting  probability distribution.

        Parameters
        ----------------------------
        init_prob: a numpy array
                  -> the initial probability distribution of the walker.

        steps: an integer
              -> the number of steps in the classical random walk.

        plot: a boolean
              -> specify whether to plot the probability distribution

        Returns
        ----------------------------
        prob: a numpy array
              -> the resulting probability distribution
        """

        coin_probs = np.array([self.coin_probs[0], self.coin_probs[1]]) / np.sum(
            self.coin_probs
        )

        init_prob = np.hstack((np.zeros(steps), init_prob, np.zeros(steps)))
        prob = init_prob
        for i in range(steps):
            prob_vec = np.vstack((self.qrw.conv(prob, 1), self.qrw.conv(prob, -1)))
            prob = (coin_probs.reshape(1, 2) @ prob_vec).flatten()

        if plot:
            plt.figure(figsize=(12, 6))
            sns.set_style("darkgrid")
            plt.gca().set_facecolor("black")
            cmap1D = matplotlib.colors.LinearSegmentedColormap.from_list(
                "", ["indigo", "blueviolet", "violet", "magenta"]
            )
            norm = plt.Normalize(
                vmin=min(prob), vmax=max(prob)
            )  # Normalize the data for colormap
            colors = [cmap1D(norm(value)) for value in prob]

            N = len(prob)
            if N % 2 == 0:
                plt.bar(range(-int(N / 2), int(N / 2)), prob, color=colors)
            else:
                plt.bar(
                    range(-int((N - 1) / 2), int((N - 1) / 2) + 1), prob, color=colors
                )

            plt.title(
                f"Classical Random Walks \n coin = [{coin_probs[0]:.3f},{coin_probs[1]:.3f}], $\\quad$ Steps = {steps}",
                fontsize=15,
            )
            plt.xlabel("Positions", fontsize=15)
            plt.ylabel("Probability", fontsize=15)
            plt.show()

        return prob


class QRW_qiskit_simulation:
    """
    simulator class for simulating 1D Quantum Random Walks with one single starting position using Qiskit AerSimulator

    """

    def __init__(self):
        pass

    def circuit(
        self,
        steps: int,
        theta: float,
        lamb: float,
        coin_state: np.ndarray,
        draw: bool = True,
    ) -> QuantumCircuit:
        """
        Given the initial coin state and the parameters of the coin flip operator, the circuit function returns the corresponding quantum
        circuit and optionally draws the circuit diagram.

        Parameters
        ----------------------
        steps: an integer
               -> the number of steps

        theta: a float
               -> the polar angle parametrizing the Ry rotation

        lamb: a float
              -> the azimuthal angle parametrizing the Rz rotation

        coin_state: a numpy array
                    -> the quantum state of the coin as a complex vector

        draw: a boolean
              -> specify whether the circuit diagram should be drawn

        Returns
        ---------------------
        circ: a quantum circuit object
              -> the resulting QRW circuit
        """

        if steps == 0:
            n = 1
        else:
            n = math.ceil(np.log2(2 * steps + 1))
        coin = QuantumRegister(1, name="coin")
        pos = QuantumRegister(n, name="position")
        cr = ClassicalRegister(n, name="classical")

        circ = QuantumCircuit(coin, cr)
        circ.add_register(pos)

        circ.initialize(coin_state, coin)
        circ.x(pos[-1])  # most significant bit at the bottom

        # Define the controlled gates sequence for addition and subtraction
        add_mcx = []
        for i in reversed(range(1, n + 1)):
            add_mcx.append(XGate().control(i))
        sub_mcx = []
        for i in range(1, n + 1):
            sub_mcx.append(XGate().control(i, ctrl_state="1" * (i - 1) + "0"))

        for k in range(steps):
            circ.u(theta, 0, lamb, coin)

            for i in range(n):
                circ.append(add_mcx[i], [coin[0]] + list(pos[: n - i]))

            for i in range(n):
                circ.append(sub_mcx[i], [coin[0]] + list(pos[: i + 1]))

            circ.barrier()

        circ.measure(pos, cr)

        if draw:
            fig = circ.draw("mpl")
            display(fig)
            plt.close()

        return circ

    def memory_circuit(
        self, coin_angles: np.ndarray, memory_angle: float, draw: bool = True
    ) -> QuantumCircuit:
        """
        Given the initial coin angles and the memory angle, the memory_circuit function returns the circuit for the QRW with memory
        and optionally draws the circuit diagram.

        Parameters
        ----------------------
        coin_angles: a numpy array
               -> the Ry rotation angles for preparing the initial coin states

        memory_angle: a float
               -> the C-Ry rotation angle, which quantifies the impact of memory and also measures the level of entanglement.

        draw: a boolean
              -> specify whether the circuit diagram should be drawn

        Returns
        ---------------------
        circ: a quantum circuit object
              -> the resulting QRW with memory circuit
        """

        steps = len(coin_angles)
        n = math.ceil(np.log2(2 * steps + 1))

        position = QuantumRegister(n, name="position")
        coin = QuantumRegister(steps, name="coin")
        cr = ClassicalRegister(n, name="classical")
        circuit = QuantumCircuit(coin, cr)
        circuit.add_register(position)

        # Initialize the coin states
        for i in range(steps):
            circuit.ry(coin_angles[i], coin[i])

        # Initialize the position to |2^(n-1)> (most significant bit is at the bottom)
        circuit.x(position[-1])

        # Define the controlled gates sequence for addition and subtraction
        add_mcx = []
        for i in reversed(range(1, n + 1)):
            add_mcx.append(XGate().control(i))
        sub_mcx = []
        for i in range(1, n + 1):
            sub_mcx.append(XGate().control(i, ctrl_state="1" * (i - 1) + "0"))

        # operate on the position and store the memory
        for step in range(steps):
            # operate on the position
            for i in range(n):
                circuit.append(add_mcx[i], [coin[step]] + list(position[: n - i]))

            for i in range(n):
                circuit.append(sub_mcx[i], [coin[step]] + list(position[: i + 1]))
            circuit.barrier()

            # store the memory in the next coin
            if step < steps - 1:
                circuit.cry(memory_angle, coin[step], coin[step + 1])
                circuit.cry(-memory_angle, coin[step], coin[step + 1], ctrl_state="0")
                circuit.barrier()

        circuit.measure(position, cr)

        if draw:
            fig = circuit.draw("mpl")
            display(fig)
            plt.close()

        return circuit

    def simulate(
        self,
        steps: int,
        theta: float,
        lamb: float,
        coin_state: np.ndarray,
        shots: int = 10000,
    ) -> np.ndarray:
        """
        Given the initial coin state and the parameters of the coin flip operator, the simulate function simulates the QRW with
        a quantum circuit and returns the resulting probability distribution through probabilistic sampling.

        Parameters
        ----------------------
        steps: an integer
               -> the number of steps

        theta: a float
               -> the polar angle parametrizing the Ry rotation

        lamb: a float
              -> the azimuthal angle parametrizing the Rz rotation

        coin_state: a numpy array
                    -> the quantum state of the coin as a complex vector

        shots: an integer
               -> number of shots for sampling

        Returns
        ---------------------
        qiskit_dist: a numpy array
                    -> the resulting probability distribution after QRW
        """

        if steps == 0:
            n = 1
        else:
            n = math.ceil(np.log2(2 * steps + 1))
        coin = QuantumRegister(1, name="coin")
        pos = QuantumRegister(n, name="position")
        cr = ClassicalRegister(n, name="classical")

        circ = QuantumCircuit(coin, cr)
        circ.add_register(pos)

        circ.initialize(coin_state, coin)
        circ.x(pos[-1])

        add_mcx = []
        for i in reversed(range(1, n + 1)):
            add_mcx.append(XGate().control(i))
        sub_mcx = []
        for i in range(1, n + 1):
            sub_mcx.append(XGate().control(i, ctrl_state="1" * (i - 1) + "0"))

        for k in range(steps):
            circ.u(theta, 0, lamb, coin)

            for i in range(n):
                circ.append(add_mcx[i], [coin[0]] + list(pos[: n - i]))

            for i in range(n):
                circ.append(sub_mcx[i], [coin[0]] + list(pos[: i + 1]))

            circ.barrier()

        circ.measure(pos, cr)

        simulator = AerSimulator()

        circ = transpile(circ, simulator)
        job = simulator.run(circ, shots=shots)
        counts = job.result().get_counts()

        if steps != 0:
            qiskit_dist = np.zeros(2 * steps + 1)
            for key, item in counts.items():
                idx = int(key, 2) - 2 ** (n - 1) + steps
                qiskit_dist[idx] = item / shots
        else:
            qiskit_dist = np.ones(1)

        return qiskit_dist

    def memory_simulate(
        self, coin_angles: np.ndarray, memory_angle: float, shots: int = 10000
    ) -> np.ndarray:
        """
        Given the initial coin angles and the memory angle, the memory_simulate function simulates the QRW with memory
        and returns the resulting probability distribution through probabilistic sampling.

        Parameters
        ----------------------
        coin_angles: a numpy array
               -> the Ry rotation angles for preparing the initial coin states

        memory_angle: a float
               -> the C-Ry rotation angle, which quantifies the impact of memory and also measures the level of entanglement.

        shots: an integer
               -> number of shots for sampling

        Returns
        ---------------------
        qiskit_dist: a numpy array
                    -> the resulting probability distribution after QRW with memory
        """

        steps = len(coin_angles)
        n = math.ceil(np.log2(2 * steps + 1))

        position = QuantumRegister(n, name="position")
        coin = QuantumRegister(steps, name="coin")
        cr = ClassicalRegister(n, name="classical")
        circuit = QuantumCircuit(coin, cr)
        circuit.add_register(position)

        # Initialize the coin states
        for i in range(steps):
            circuit.ry(coin_angles[i], coin[i])

        # Initialize the position to |2^(n-1)> (most significant bit is at the bottom)
        circuit.x(position[-1])

        # Define the controlled gates sequence for addition and subtraction
        add_mcx = []
        for i in reversed(range(1, n + 1)):
            add_mcx.append(XGate().control(i))
        sub_mcx = []
        for i in range(1, n + 1):
            sub_mcx.append(XGate().control(i, ctrl_state="1" * (i - 1) + "0"))

        # operate on the position and store the memory
        for step in range(steps):
            # operate on the position
            for i in range(n):
                circuit.append(add_mcx[i], [coin[step]] + list(position[: n - i]))

            for i in range(n):
                circuit.append(sub_mcx[i], [coin[step]] + list(position[: i + 1]))
            circuit.barrier()

            # store the memory in the next coin
            if step < steps - 1:
                circuit.cry(memory_angle, coin[step], coin[step + 1])
                circuit.cry(-memory_angle, coin[step], coin[step + 1], ctrl_state="0")
                circuit.barrier()

        circuit.measure(position, cr)

        simulator = AerSimulator()

        circ = transpile(circuit, simulator)
        job = simulator.run(circ, shots=shots)
        counts = job.result().get_counts()

        qiskit_dist = np.zeros(2 * steps + 1)
        for key, item in counts.items():
            idx = int(key, 2) - 2 ** (n - 1) + steps
            qiskit_dist[idx] = item / shots

        return qiskit_dist


class harmonic_QRW_simulation:
    """
    simulator class for 1D harmonic QRW, which incorporates a second coin flip operation applied periodically.
    """

    def __init__(self):
        self.simulator = QRW_simulation(1)

    def simulate(
        self,
        coin_state: np.ndarray,
        U: List[np.ndarray],
        init_amp: np.ndarray,
        steps: int,
        period: int,
        return_amp: bool = False,
    ) -> np.ndarray:
        """
        Given an initial amplitufde distribution (1D) and relevant quantum random walks parameters, the simulate function simulates the process of
        the harmonic QRW, and returns the resulting  probability distribution or amplitude distribution if specified.

        Parameters
        ----------------------------
        coin_state: a numpy array
                   -> the quantum state of the coin as a complex vector.

        U: a list of numpy array
           -> the unitary matrices of the fundamental and the harmonic coin flip operations.

        init_amp: a numpy array
                  -> the initial amplitude distribution of the walker.

        steps: an integer
              -> the number of steps in the QRW.

        period: an integer
                -> the period of the harmonic coin flip operation relative the fundamental period, which is unity.

        return_amp: a boolean
                    -> specify if the output should be the amplitude distribution.

        Returns
        ----------------------------
        if return_amp=False ->
        prob: a numpy array
              -> the resulting probability distribution

        if return_amp=True ->
        tensor: a numpy array
                -> the resulting amplitude distribution
        """

        coin_vec = np.array(coin_state)

        tensor = np.tensordot(
            np.array(coin_vec).reshape(len(coin_vec), 1),
            init_amp.reshape((1,) + init_amp.shape),
            axes=1,
        )

        for step in range(steps):
            if (step + 1) % period == 0:
                tensor = U[1] @ U[0] @ tensor
            else:
                tensor = U[0] @ tensor

            tensor = np.vstack(
                (
                    self.simulator.conv(tensor[0, :], 1),
                    self.simulator.conv(tensor[1, :], -1),
                )
            )

        if return_amp == True:
            return tensor
        else:
            prob = np.tensordot(
                np.ones((1, tensor.shape[0])),
                (tensor * np.conjugate(tensor)).real,
                axes=1,
            ).reshape(init_amp.shape)
            return prob


class qutrit_QRW_simulation:
    """
    simulator class for 1D QRW with qutrit, analogous to a 3-sided coin random walk.
    """

    def __init__(self):
        self.simulator = QRW_simulation(1)

    def simulate(
        self,
        coin_state: np.ndarray,
        U: np.ndarray,
        init_amp: np.ndarray,
        steps: int,
        return_amp: bool = False,
    ) -> np.ndarray:
        """
        Given an initial amplitufde distribution (1D) and relevant quantum random walks parameters, the simulate function simulates the process of
        the QRW with qutrit, and returns the resulting  probability distribution or amplitude distribution if specified.

        Parameters
        ----------------------------
        coin_state: a numpy array
                   -> the quantum state of the coin as a complex vector.

        U: a numpy array
           -> the unitary matrices of the coin flip operation.

        init_amp: a numpy array
                  -> the initial amplitude distribution of the walker.

        steps: an integer
              -> the number of steps in the QRW.

        return_amp: a boolean
                    -> specify if the output should be the amplitude distribution.

        Returns
        ----------------------------
        if return_amp=False ->
        prob: a numpy array
              -> the resulting probability distribution

        if return_amp=True ->
        tensor: a numpy array
                -> the resulting amplitude distribution
        """

        coin_vec = np.array(coin_state)

        tensor = np.tensordot(
            np.array(coin_vec).reshape(len(coin_vec), 1),
            init_amp.reshape((1,) + init_amp.shape),
            axes=1,
        )

        for step in range(steps):
            tensor = U @ tensor

            tensor = np.vstack(
                (
                    self.simulator.conv(tensor[0, :], 1),
                    tensor[1, :],
                    self.simulator.conv(tensor[2, :], -1),
                )
            )

        if return_amp == True:
            return tensor
        else:
            prob = np.tensordot(
                np.ones((1, tensor.shape[0])),
                (tensor * np.conjugate(tensor)).real,
                axes=1,
            ).reshape(init_amp.shape)
            return prob


class QRW_visualization:
    """
    visualization class for visualizing 1D and 2D QRW interactively.
    """

    def __init__(
        self,
        init_amp: np.ndarray,
        max_steps: int,
        dim: int,
        model: str = "basic",
        plot_type: str = "time_slice",
    ):
        """
        Parameters
        ----------------------
        init_amp: a numpy array
                  -> the initial amplitude distribution.

        max_steps: an integer
                   -> the maximum allowed number of steps in the visualization.

        dim: an integer
             -> the dimension of the QRW

        model: a string
               -> choose between 'basic' (one operation), 'harmonic' (2 operations) or 'qutrit' QRW model for visualization.
                  For 2D QRW, only  'basic' is available

        plot_type: a string
                   -> the type of plot. Available options for 1D QRW are "time_slice" and "timeline". For 2D QRW, only "time_slice"
        """

        if len(init_amp.shape) != dim:
            raise ValueError(
                "the dimension of init_amp does not match the specify dim."
            )
        if dim not in [1, 2]:
            raise ValueError("dim must be either 1 or 2.")
        if model not in ["basic", "harmonic", "qutrit"]:
            raise ValueError("available models are 'basic','harmonic',or 'qutrit'.")
        if dim == 1:
            if plot_type not in ["time_slice", "timeline"]:
                raise ValueError("plot_type is either 'time_slice' or 'timeline'.")
        else:
            if model != "basic":
                raise ValueError("2D QRW only offers the 'basic' model.")
            if plot_type != "time_slice":
                raise ValueError("2D QRW only offers the 'time_slice' option.")

        self.max_steps = max_steps

        if dim == 1:
            self.init_amp = np.hstack(
                (np.zeros(max_steps), init_amp, np.zeros(max_steps))
            )
        else:
            length = len(init_amp) + 2 * max_steps
            modified = np.zeros((length, length))
            modified[max_steps:-max_steps, max_steps:-max_steps] = init_amp
            self.init_amp = modified

        if len(self.init_amp) % 2 != 0:
            self.bound = int((len(self.init_amp) - 1) / 2)
        else:
            self.bound = int(len(self.init_amp) / 2)

        self.dim = dim
        self.model = model
        self.plot_type = plot_type
        if model == "basic":
            self.simulator = QRW_simulation(dim)
        if model == "harmonic":
            self.simulator = harmonic_QRW_simulation()
        if model == "qutrit":
            self.simulator = qutrit_QRW_simulation()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the input distribution such that the highest probability is mapped to 1 whereas the lowest is mapped to 0.
        """
        return (x - np.min(x)) / np.max(x)

    def plot(self) -> None:
        """
        Make interactive plots of 1D or 2D QRW.
        """

        sns.set_style("darkgrid")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["k", "rebeccapurple", "blueviolet", "mediumorchid", "magenta"]
        )

        init_amp = self.init_amp
        max_steps = self.max_steps

        if self.dim == 1:
            if self.plot_type == "time_slice":  # 1D QRW time_slice plot
                if self.model == "basic":

                    def mono_time_slice_1D(steps, theta, lamb, a, phi):
                        # Define the coin operator U based on theta and lambda
                        U = np.array(
                            [
                                [
                                    np.cos(theta / 2),
                                    -np.exp(lamb * 1j) * np.sin(theta / 2),
                                ],
                                [
                                    np.sin(theta / 2),
                                    np.exp(lamb * 1j) * np.cos(theta / 2),
                                ],
                            ]
                        )

                        # Define the coin state
                        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])

                        # Run the QRW simulation
                        result = self.simulator.simulate(
                            [coin_state], [U], self.init_amp, steps
                        )

                        # Plot the result
                        plt.clf()
                        plt.figure(figsize=(12, 6))
                        plt.gca().set_facecolor("black")
                        cmap1D = matplotlib.colors.LinearSegmentedColormap.from_list(
                            "", ["indigo", "blueviolet", "violet", "magenta"]
                        )
                        norm = plt.Normalize(vmin=min(result), vmax=max(result))
                        colors = [cmap1D(norm(value)) for value in result]

                        if len(init_amp) % 2 != 0:
                            plt.bar(
                                range(-self.bound, self.bound + 1), result, color=colors
                            )
                        else:
                            plt.bar(
                                range(-self.bound, self.bound), result, color=colors
                            )

                        plt.xticks(rotation=60)
                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Probability", fontsize=15)
                        plt.title(
                            f"Quantum Random Walks $\\quad$ Steps = {steps} \n θ = {round(theta / np.pi, 3)}π $\\quad$ λ = {round(lamb / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        mono_time_slice_1D,
                        steps=widgets.IntSlider(
                            value=0, min=0, max=max_steps, step=1, description="Steps"
                        ),
                        theta=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=np.pi,
                            step=0.05 * np.pi,
                            description="θ",
                        ),
                        lamb=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ",
                        ),
                        a=widgets.FloatSlider(
                            value=0, min=0, max=1, step=0.05, description="a"
                        ),
                        phi=widgets.FloatSlider(
                            value=0,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="ϕ",
                        ),
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

                if self.model == "harmonic":  # 1D harmonic QRW time_slice plot

                    def harmonic_time_slice_1D(
                        steps, theta1, lamb1, theta2, lamb2, a, phi, period
                    ):
                        U1 = np.array(
                            [
                                [
                                    np.cos(theta1 / 2),
                                    -np.exp(lamb1 * 1j) * np.sin(theta1 / 2),
                                ],
                                [
                                    np.sin(theta1 / 2),
                                    np.exp(lamb1 * 1j) * np.cos(theta1 / 2),
                                ],
                            ]
                        )
                        U2 = np.array(
                            [
                                [
                                    np.cos(theta2 / 2),
                                    -np.exp(lamb2 * 1j) * np.sin(theta2 / 2),
                                ],
                                [
                                    np.sin(theta2 / 2),
                                    np.exp(lamb2 * 1j) * np.cos(theta2 / 2),
                                ],
                            ]
                        )

                        # Define the coin state
                        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])

                        # Run the QRW simulation
                        result = self.simulator.simulate(
                            coin_state, [U1, U2], self.init_amp, steps, period
                        )

                        # Plot the result
                        plt.clf()
                        plt.figure(figsize=(12, 6))
                        plt.gca().set_facecolor("black")
                        cmap1D = matplotlib.colors.LinearSegmentedColormap.from_list(
                            "", ["indigo", "blueviolet", "violet", "magenta"]
                        )
                        norm = plt.Normalize(vmin=min(result), vmax=max(result))
                        colors = [cmap1D(norm(value)) for value in result]
                        if len(init_amp) % 2 != 0:
                            plt.bar(
                                range(-self.bound, self.bound + 1), result, color=colors
                            )
                        else:
                            plt.bar(
                                range(-self.bound, self.bound), result, color=colors
                            )
                        plt.xticks(rotation=60)
                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Probability", fontsize=15)
                        plt.title(
                            f"Harmonic Quantum Random Walks $\\quad$ Steps = {steps} \n θ1 = {round(theta1 / np.pi, 3)}π $\\quad$ λ1 = {round(lamb1 / np.pi, 3)}π $\\quad$ θ2 = {round(theta2 / np.pi, 3)}π $\\quad$ λ2 = {round(lamb2 / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)} $\\quad$ period = {period}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        harmonic_time_slice_1D,
                        steps=widgets.IntSlider(
                            value=0, min=0, max=max_steps, step=1, description="Steps"
                        ),
                        theta1=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=np.pi,
                            step=0.05 * np.pi,
                            description="θ1",
                        ),
                        lamb1=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ1",
                        ),
                        theta2=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=np.pi,
                            step=0.05 * np.pi,
                            description="θ2",
                        ),
                        lamb2=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ2",
                        ),
                        a=widgets.FloatSlider(
                            value=0, min=0, max=1, step=0.05, description="a"
                        ),
                        phi=widgets.FloatSlider(
                            value=0,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="ϕ",
                        ),
                        period=widgets.IntSlider(
                            value=1,
                            min=1,
                            max=max_steps + 1,
                            step=1,
                            description="period",
                        ),
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

                if self.model == "qutrit":  # 1d qutrit QRW time_slice plot
                    Steps_slid = widgets.IntSlider(
                        value=0, min=0, max=max_steps, step=1, description="Steps"
                    )
                    theta_slid = widgets.FloatSlider(
                        value=np.pi / 2,
                        min=0,
                        max=np.pi,
                        step=0.05 * np.pi,
                        description="θ",
                    )
                    lamb_slid = widgets.FloatSlider(
                        value=np.pi,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="λ",
                    )
                    a_slid = widgets.FloatSlider(
                        value=0, min=0, max=1, step=0.05, description="a"
                    )
                    b_slid = widgets.FloatSlider(
                        value=0, min=0, max=1, step=0.05, description="b"
                    )
                    phi1_slid = widgets.FloatSlider(
                        value=0,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="ϕ1",
                    )
                    phi2_slid = widgets.FloatSlider(
                        value=0,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="ϕ2",
                    )

                    def a_b_change(change):
                        b_slid.max = (1 - a_slid.value**2) ** 0.5

                    a_slid.observe(a_b_change, names="value")

                    def qutrit_time_slice_1D(steps, theta, lamb, a, b, phi1, phi2):
                        # Define the coin operator U based on theta and lambda
                        U = np.array(
                            [
                                [
                                    np.exp(lamb * 1j) * np.cos(theta / 2) ** 2,
                                    1 / 2**0.5 * np.sin(theta),
                                    np.exp(-lamb * 1j) * np.sin(theta / 2) ** 2,
                                ],
                                [
                                    -1 / 2**0.5 * np.exp(lamb * 1j) * np.sin(theta),
                                    np.cos(theta),
                                    1 / 2**0.5 * np.exp(-lamb * 1j) * np.sin(theta),
                                ],
                                [
                                    np.exp(lamb * 1j) * np.sin(theta / 2) ** 2,
                                    -1 / 2**0.5 * np.sin(theta),
                                    np.exp(-lamb * 1j) * np.cos(theta / 2) ** 2,
                                ],
                            ]
                        )

                        # Define the coin state
                        coin_state = np.array(
                            [
                                a,
                                b * np.exp(1j * phi1),
                                (1 - a**2 - b**2) ** 0.5 * np.exp(1j * phi2),
                            ]
                        )

                        # Run the QRW simulation
                        result = self.simulator.simulate(
                            coin_state, U, self.init_amp, steps
                        )

                        # Plot the result
                        plt.clf()
                        plt.figure(figsize=(12, 6))
                        plt.gca().set_facecolor("black")
                        cmap1D = matplotlib.colors.LinearSegmentedColormap.from_list(
                            "", ["indigo", "blueviolet", "violet", "magenta"]
                        )
                        norm = plt.Normalize(vmin=min(result), vmax=max(result))
                        colors = [cmap1D(norm(value)) for value in result]
                        if len(init_amp) % 2 != 0:
                            plt.bar(
                                range(-self.bound, self.bound + 1), result, color=colors
                            )
                        else:
                            plt.bar(
                                range(-self.bound, self.bound), result, color=colors
                            )
                        plt.xticks(rotation=60)
                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Probability", fontsize=15)
                        plt.title(
                            f"Quantum Random Walks with Qutrit $\\quad$ Steps = {steps} \n θ = {round(theta / np.pi, 3)}π $\\quad$ λ = {round(lamb / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        qutrit_time_slice_1D,
                        steps=Steps_slid,
                        theta=theta_slid,
                        lamb=lamb_slid,
                        a=a_slid,
                        b=b_slid,
                        phi1=phi1_slid,
                        phi2=phi2_slid,
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

            else:  # 1D QRW timeline plot
                if self.model == "basic":  # 1D basic QRW timeline plot

                    def mono_timeline_1D(steps, theta, lamb, a, phi):
                        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])

                        U = np.array(
                            [
                                [
                                    np.cos(theta / 2),
                                    -np.exp(lamb * 1j) * np.sin(theta / 2),
                                ],
                                [
                                    np.sin(theta / 2),
                                    np.exp(lamb * 1j) * np.cos(theta / 2),
                                ],
                            ]
                        )

                        evolution = np.array(
                            [
                                self.simulator.simulate([coin_state], [U], init_amp, s)
                                for s in range(steps)
                            ]
                        )
                        evolution = np.array(
                            [self.normalize(evolution[i, :]) for i in range(steps)]
                        )

                        if steps == 0:
                            evolution = init_amp.reshape(1, len(init_amp))

                        if steps > 50:
                            x = 14 * len(init_amp) / 100
                            y = 2 + 12 / 100 * steps
                        else:
                            x = 14 * len(init_amp) / 100
                            y = 6

                        plt.clf()
                        plt.figure(figsize=(x, y))

                        sns.heatmap(evolution, cmap=cmap)

                        xmin, xmax = plt.gca().get_xlim()

                        if len(init_amp) % 2 != 0:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound + 1, 5)
                            ]
                        else:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound, 5)
                            ]
                        ylabels = [f"{int(5 * i)}" for i in range(steps // 5 + 1)]

                        plt.xticks(
                            np.arange(xmin, xmax + 1, 5), labels=xlabels, fontsize=15
                        )
                        plt.yticks(
                            list(range(0, 5 * (steps // 5) + 1, 5)),
                            labels=ylabels,
                            fontsize=15,
                        )

                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Steps", fontsize=15)
                        plt.title(
                            f"Quantum Random Walks \n θ = {round(theta / np.pi, 3)}π $\\quad$ λ = {round(lamb / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        mono_timeline_1D,
                        steps=widgets.IntSlider(
                            value=0, min=0, max=max_steps, step=1, description="Steps"
                        ),
                        theta=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="θ",
                        ),
                        lamb=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ",
                        ),
                        a=widgets.FloatSlider(
                            value=0, min=0, max=1, step=0.05, description="a"
                        ),
                        phi=widgets.FloatSlider(
                            value=0,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="ϕ",
                        ),
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

                if self.model == "harmonic":  # 1D harmonic QRW timeline plot

                    def harmonic_timeline_1D(
                        steps, theta1, lamb1, theta2, lamb2, a, phi, period
                    ):
                        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])

                        U1 = np.array(
                            [
                                [
                                    np.cos(theta1 / 2),
                                    -np.exp(lamb1 * 1j) * np.sin(theta1 / 2),
                                ],
                                [
                                    np.sin(theta1 / 2),
                                    np.exp(lamb1 * 1j) * np.cos(theta1 / 2),
                                ],
                            ]
                        )
                        U2 = np.array(
                            [
                                [
                                    np.cos(theta2 / 2),
                                    -np.exp(lamb2 * 1j) * np.sin(theta2 / 2),
                                ],
                                [
                                    np.sin(theta2 / 2),
                                    np.exp(lamb2 * 1j) * np.cos(theta2 / 2),
                                ],
                            ]
                        )

                        evolution = np.array(
                            [
                                self.simulator.simulate(
                                    coin_state, [U1, U2], init_amp, s, period
                                )
                                for s in range(steps)
                            ]
                        )
                        evolution = np.array(
                            [self.normalize(evolution[i, :]) for i in range(steps)]
                        )

                        if steps == 0:
                            evolution = init_amp.reshape(1, len(init_amp))

                        if steps > 50:
                            x = 14 * len(init_amp) / 100
                            y = 2 + 12 / 100 * steps
                        else:
                            x = 14 * len(init_amp) / 100
                            y = 6

                        plt.clf()
                        plt.figure(figsize=(x, y))

                        sns.heatmap(evolution, cmap=cmap)

                        xmin, xmax = plt.gca().get_xlim()

                        if len(init_amp) % 2 != 0:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound + 1, 5)
                            ]
                        else:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound, 5)
                            ]
                        ylabels = [f"{int(5 * i)}" for i in range(steps // 5 + 1)]

                        plt.xticks(
                            np.arange(xmin, xmax + 1, 5), labels=xlabels, fontsize=15
                        )
                        plt.yticks(
                            list(range(0, 5 * (steps // 5) + 1, 5)),
                            labels=ylabels,
                            fontsize=15,
                        )

                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Steps", fontsize=15)
                        plt.title(
                            f"Harmonic Quantum Random Walks \n θ1 = {round(theta1 / np.pi, 3)}π $\\quad$ λ1 = {round(lamb1 / np.pi, 3)}π $\\quad$ θ2 = {round(theta2 / np.pi, 3)}π $\\quad$ λ2 = {round(lamb2 / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)} $\\quad$ period = {period}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        harmonic_timeline_1D,
                        steps=widgets.IntSlider(
                            value=0, min=0, max=max_steps, step=1, description="Steps"
                        ),
                        theta1=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="θ1",
                        ),
                        lamb1=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ1",
                        ),
                        theta2=widgets.FloatSlider(
                            value=np.pi / 2,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="θ2",
                        ),
                        lamb2=widgets.FloatSlider(
                            value=np.pi,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="λ2",
                        ),
                        a=widgets.FloatSlider(
                            value=0, min=0, max=1, step=0.05, description="a"
                        ),
                        phi=widgets.FloatSlider(
                            value=0,
                            min=0,
                            max=2 * np.pi,
                            step=0.05 * np.pi,
                            description="ϕ",
                        ),
                        period=widgets.IntSlider(
                            value=1,
                            min=1,
                            max=max_steps + 1,
                            step=1,
                            description="period",
                        ),
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

                if self.model == "qutrit":  # 1D qutrit QRW timeline plot
                    Steps_slid = widgets.IntSlider(
                        value=0, min=0, max=max_steps, step=1, description="Steps"
                    )
                    theta_slid = widgets.FloatSlider(
                        value=np.pi / 2,
                        min=0,
                        max=np.pi,
                        step=0.05 * np.pi,
                        description="θ",
                    )
                    lamb_slid = widgets.FloatSlider(
                        value=np.pi,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="λ",
                    )
                    a_slid = widgets.FloatSlider(
                        value=0, min=0, max=1, step=0.05, description="a"
                    )
                    b_slid = widgets.FloatSlider(
                        value=0, min=0, max=1, step=0.05, description="b"
                    )
                    phi1_slid = widgets.FloatSlider(
                        value=0,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="ϕ1",
                    )
                    phi2_slid = widgets.FloatSlider(
                        value=0,
                        min=0,
                        max=2 * np.pi,
                        step=0.05 * np.pi,
                        description="ϕ2",
                    )

                    def qutrit_timeline_1D(steps, theta, lamb, a, b, phi1, phi2):
                        coin_state = np.array(
                            [
                                a,
                                b * np.exp(1j * phi1),
                                (1 - a**2 - b**2) ** 0.5 * np.exp(1j * phi2),
                            ]
                        )

                        U = np.array(
                            [
                                [
                                    np.exp(lamb * 1j) * np.cos(theta / 2) ** 2,
                                    1 / 2**0.5 * np.sin(theta),
                                    np.exp(-lamb * 1j) * np.sin(theta / 2) ** 2,
                                ],
                                [
                                    -1 / 2**0.5 * np.exp(lamb * 1j) * np.sin(theta),
                                    np.cos(theta),
                                    1 / 2**0.5 * np.exp(-lamb * 1j) * np.sin(theta),
                                ],
                                [
                                    np.exp(lamb * 1j) * np.sin(theta / 2) ** 2,
                                    -1 / 2**0.5 * np.sin(theta),
                                    np.exp(-lamb * 1j) * np.cos(theta / 2) ** 2,
                                ],
                            ]
                        )

                        evolution = np.array(
                            [
                                self.simulator.simulate(coin_state, U, init_amp, s)
                                for s in range(steps)
                            ]
                        )
                        evolution = np.array(
                            [self.normalize(evolution[i, :]) for i in range(steps)]
                        )

                        if steps == 0:
                            evolution = init_amp.reshape(1, len(init_amp))

                        if steps > 50:
                            x = 14 * len(init_amp) / 100
                            y = 2 + 12 / 100 * steps
                        else:
                            x = 14 * len(init_amp) / 100
                            y = 6

                        plt.clf()
                        plt.figure(figsize=(x, y))

                        sns.heatmap(evolution, cmap=cmap)

                        xmin, xmax = plt.gca().get_xlim()

                        if len(init_amp) % 2 != 0:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound + 1, 5)
                            ]
                        else:
                            xlabels = [
                                f"{int(x)}"
                                for x in np.arange(-self.bound, self.bound, 5)
                            ]
                        ylabels = [f"{int(5 * i)}" for i in range(steps // 5 + 1)]

                        plt.xticks(
                            np.arange(xmin, xmax + 1, 5), labels=xlabels, fontsize=15
                        )
                        plt.yticks(
                            list(range(0, 5 * (steps // 5) + 1, 5)),
                            labels=ylabels,
                            fontsize=15,
                        )

                        plt.xlabel("Positions", fontsize=15)
                        plt.ylabel("Steps", fontsize=15)
                        plt.title(
                            f"Quantum Random Walks with Qutrit \n θ = {round(theta / np.pi, 3)}π $\\quad$ λ = {round(lamb / np.pi, 3)}π $\\quad$ coin state = {np.round(coin_state, 3)}",
                            fontsize=15,
                        )
                        plt.show()

                    widget = interactive(
                        qutrit_timeline_1D,
                        steps=Steps_slid,
                        theta=theta_slid,
                        lamb=lamb_slid,
                        a=a_slid,
                        b=b_slid,
                        phi1=phi1_slid,
                        phi2=phi2_slid,
                    )

                    controls = HBox(
                        widget.children[:-1], layout=Layout(flex_flow="row wrap")
                    )
                    output = widget.children[-1]
                    display(VBox([controls, output]))

        else:  # 2D QRW time_slice plot
            if init_amp.shape[0] != init_amp.shape[1]:
                raise ValueError("init_amp must be a N x N 2D array")

            def time_slice_2D(steps, theta1, lamb1, theta2, lamb2, a1, phi1, a2, phi2):
                coin_state = np.array(
                    [
                        [a1, (1 - a1**2) ** 0.5 * np.exp(1j * phi1)],
                        [a2, (1 - a2**2) ** 0.5 * np.exp(1j * phi2)],
                    ]
                )

                U1 = np.array(
                    [
                        [np.cos(theta1 / 2), -np.exp(lamb1 * 1j) * np.sin(theta1 / 2)],
                        [np.sin(theta1 / 2), np.exp(lamb1 * 1j) * np.cos(theta1 / 2)],
                    ]
                )
                U2 = np.array(
                    [
                        [np.cos(theta2 / 2), -np.exp(lamb2 * 1j) * np.sin(theta2 / 2)],
                        [np.sin(theta2 / 2), np.exp(lamb2 * 1j) * np.cos(theta2 / 2)],
                    ]
                )
                U = [U1, U2]

                plt.clf()
                plt.figure(figsize=(10, 8))
                ppp = self.simulator.simulate(coin_state, U, init_amp, steps)
                if steps == 0:
                    ax = sns.heatmap(init_amp, cmap=cmap)
                else:
                    ax = sns.heatmap(
                        (ppp - np.min(ppp)) / (np.max(ppp) - np.min(ppp)), cmap=cmap
                    )

                if len(init_amp) % 2 != 0:
                    ax.set_xticks(
                        range(0, len(init_amp), 5),
                        labels=[f"{i}" for i in range(-self.bound, self.bound + 1, 5)],
                    )
                    ax.set_yticks(
                        range(0, len(init_amp), 5),
                        labels=[f"{i}" for i in range(self.bound, -self.bound - 1, -5)],
                    )
                else:
                    ax.set_xticks(
                        range(0, len(init_amp), 5),
                        labels=[f"{i}" for i in range(-self.bound, self.bound, 5)],
                    )
                    ax.set_yticks(
                        range(0, len(init_amp), 5),
                        labels=[f"{i}" for i in range(self.bound, -self.bound, -5)],
                    )
                plt.show()

            widget = interactive(
                time_slice_2D,
                steps=widgets.IntSlider(
                    value=0, min=0, max=max_steps, step=1, description="Steps"
                ),
                theta1=widgets.FloatSlider(
                    value=np.pi / 2,
                    min=0,
                    max=2 * np.pi,
                    step=0.05 * np.pi,
                    description="θ1",
                ),
                lamb1=widgets.FloatSlider(
                    value=np.pi,
                    min=0,
                    max=2 * np.pi,
                    step=0.05 * np.pi,
                    description="λ1",
                ),
                theta2=widgets.FloatSlider(
                    value=np.pi / 2,
                    min=0,
                    max=2 * np.pi,
                    step=0.05 * np.pi,
                    description="θ2",
                ),
                lamb2=widgets.FloatSlider(
                    value=np.pi,
                    min=0,
                    max=2 * np.pi,
                    step=0.05 * np.pi,
                    description="λ2",
                ),
                a1=widgets.FloatSlider(
                    value=0, min=0, max=1, step=0.05, description="a1"
                ),
                phi1=widgets.FloatSlider(
                    value=0, min=0, max=2 * np.pi, step=0.05 * np.pi, description="ϕ1"
                ),
                a2=widgets.FloatSlider(
                    value=0, min=0, max=1, step=0.05, description="a2"
                ),
                phi2=widgets.FloatSlider(
                    value=0, min=0, max=2 * np.pi, step=0.05 * np.pi, description="ϕ2"
                ),
            )

            controls = HBox(widget.children[:-1], layout=Layout(flex_flow="row wrap"))
            output = widget.children[-1]
            display(VBox([controls, output]))


class QRW_optimization:
    """
    Optimization class for fitting 1D probability distribution with QRW (basic) or hamonic QRW (harmonic).
    """

    def __init__(self, end_prob: np.ndarray, model: str = "basic"):
        """
        Parameters
        ----------------------
        end_prob: a numpy array
                  -> the target probability distribution to be fitted

        model: a string
               -> the QRW model used to fit the distribution. Available options are "basic" or "harmonic"
        """

        self.end_prob = QRW_simulation(1).insert_zero(end_prob)
        self.max_steps = int((len(self.end_prob) - 1) / 2)
        self.model = model
        if model == "basic":
            self.simulator = QRW_simulation(1)
        else:
            self.simulator = harmonic_QRW_simulation()

    def objective_mono(self, x: np.ndarray, *args) -> float:
        """
        Objective function which evaluates the total squared error of the basic QRW distribution, given the relevant parameters,
        relative to the target distribution.

        Parameters
        ----------------------
        x: a numpy array
           -> the array contains the relevant parameters for QRW. Namely, [start,steps,theta,lamb,a,phi], where start is the initial position

        Returns
        ---------------------
        loss: a float
              -> the resulting total squared error
        """

        end_prob = args
        start, steps, theta, lamb, a, phi = x
        init_amp = np.array(
            [
                1 if i == int(start) else 0
                for i in range(-self.max_steps, self.max_steps + 1)
            ]
        )

        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])
        U = np.array(
            [
                [np.cos(theta / 2), -np.exp(lamb * 1j) * np.sin(theta / 2)],
                [np.sin(theta / 2), np.exp(lamb * 1j) * np.cos(theta / 2)],
            ]
        )

        steps = min(int(steps), self.max_steps - int(abs(start)))
        fit_prob = self.simulator.simulate([coin_state], [U], init_amp, steps)

        loss = np.sum((fit_prob - end_prob) ** 2)

        return loss

    def objective_harmonic(self, x: np.ndarray, *args) -> float:
        """
        Objective function which evaluates the total squared error of the harmonic QRW distribution, given the relevant parameters,
        relative to the target distribution.

        Parameters
        ----------------------
        x: a numpy array
           -> the array contains the relevant parameters for QRW. Namely, [start,steps,theta1,lamb1,theta2,lamb2,a,phi.period], where start is the initial position

        Returns
        ---------------------
        loss: a float
              -> the resulting total squared error
        """

        end_prob = args
        start, steps, theta1, lamb1, theta2, lamb2, a, phi, period = x
        init_amp = np.array(
            [
                1 if i == int(start) else 0
                for i in range(-self.max_steps, self.max_steps + 1)
            ]
        )

        coin_state = np.array([a, (1 - a**2) ** 0.5 * np.exp(1j * phi)])

        U1 = np.array(
            [
                [np.cos(theta1 / 2), -np.exp(lamb1 * 1j) * np.sin(theta1 / 2)],
                [np.sin(theta1 / 2), np.exp(lamb1 * 1j) * np.cos(theta1 / 2)],
            ]
        )
        U2 = np.array(
            [
                [np.cos(theta2 / 2), -np.exp(lamb2 * 1j) * np.sin(theta2 / 2)],
                [np.sin(theta2 / 2), np.exp(lamb2 * 1j) * np.cos(theta2 / 2)],
            ]
        )

        steps = min(int(steps), self.max_steps - int(abs(start)))
        period = min(int(period), steps + 1)

        fit_prob = self.simulator.simulate(
            coin_state, [U1, U2], init_amp, steps, period
        )

        loss = np.sum((fit_prob - end_prob) ** 2)

        return loss

    def optimize(
        self, rounds: int = 500, plot: bool = True
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Minimize the objective function to fit the target distribution. Optionally, the barplot comparing the fitted and the target
        distributions will be plotted.

        Parameters
        ----------------------
        rounds: an integer
                -> the number of optimization rounds. The best solution will be selected from all the optimization rounds

        plot: a boolean
           -> specify whether a comparative barplot should be drawn

        Returns
        ---------------------
        output: a dictionary
              -> the dictionary contains the best fitted parameters

        fit_prob: a numpy array
                  -> the fitted probability distribution
        """

        if self.model == "basic":
            bounds = [
                (-self.max_steps, self.max_steps + 0.5),
                (0, self.max_steps + 0.5),
                (0, np.pi),
                (0, 2 * np.pi),
                (0, 1),
                (0, 2 * np.pi),
            ]

            loss = []
            sols = []
            for i in range(rounds):
                start0 = np.random.uniform(-self.max_steps, self.max_steps + 0.5)
                x0 = np.array(
                    [self.max_steps + 0.5, np.pi, 2 * np.pi, 1, 2 * np.pi]
                ) * np.random.rand(5)
                x0 = np.hstack((start0, x0))
                opt = minimize(
                    self.objective_mono,
                    x0,
                    args=(self.end_prob),
                    method="SLSQP",
                    bounds=bounds,
                    tol=1e-8,
                    options={"maxiter": 200000},
                )

                sols.append(opt.x)
                loss.append(self.objective_mono(opt.x, self.end_prob))

            best_sol = sols[np.argmin(loss)]

            start1, steps1, theta1, lamb1, a1, phi1 = best_sol

            U_sol = np.array(
                [
                    [np.cos(theta1 / 2), -np.exp(lamb1 * 1j) * np.sin(theta1 / 2)],
                    [np.sin(theta1 / 2), np.exp(lamb1 * 1j) * np.cos(theta1 / 2)],
                ]
            )

            init_amp = np.array(
                [
                    1 if i == int(start1) else 0
                    for i in range(-self.max_steps, self.max_steps + 1)
                ]
            )

            steps1 = min(int(steps1), self.max_steps - int(abs(start1)))

            coin_state1 = np.array([a1, (1 - a1**2) ** 0.5 * np.exp(1j * phi1)])

            fit_prob = self.simulator.simulate([coin_state1], [U_sol], init_amp, steps1)

            n = max(
                len(np.where(self.end_prob != 0)[0]), len(np.where(fit_prob != 0)[0])
            )

            best_loss = np.min(loss) / n

            print(
                f"MSE: {best_loss}, std : {np.std(loss) / n}, x0: {int(start1)}, steps: {steps1}, θ: {theta1:.3f}, λ: {lamb1:.3f}, a: {a1:.3f}, ϕ: {phi1:.3f}"
            )

            output = {
                "x0": int(start1),
                "steps": steps1,
                "θ": theta1,
                "λ": lamb1,
                "a": a1,
                "ϕ": phi1,
            }

            if plot:
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_axes([0, 0, 1, 1])
                sns.set_style("darkgrid")
                plt.gca().set_facecolor("black")

                ax.bar(
                    range(-self.max_steps, self.max_steps + 1),
                    fit_prob,
                    label="fitter",
                    width=0.7 * (0.5 + 0.01 * self.max_steps),
                    align="edge",
                    color="rebeccapurple",
                )
                ax.bar(
                    range(-self.max_steps, self.max_steps + 1),
                    self.end_prob,
                    label="target",
                    width=-0.5 * (0.5 + 0.01 * self.max_steps),
                    align="edge",
                    color="magenta",
                )
                ax.bar(
                    [0],
                    [0],
                    label=f"MSE = {1000 * best_loss:.5f}e-3",
                    color="white",
                    alpha=0.01,
                )
                ax.set_xlabel("Positions", fontsize=15)
                ax.set_ylabel("Probability", fontsize=15)
                ax.set_title(
                    f"Quantum Random Walks Fitter \n $x_{0}$={int(start1)},  Steps={steps1},  θ={round(theta1, 3)},  λ={round(lamb1, 3)},  a={round(a1, 3)},  ϕ={round(phi1, 3)}",
                    fontsize=15,
                )
                ax.legend(fontsize=15)
                plt.show()

        else:  # harmonic QRW model
            bounds = [
                (-self.max_steps, self.max_steps + 0.5),
                (0, self.max_steps + 0.5),
                (0, np.pi),
                (0, 2 * np.pi),
                (0, np.pi),
                (0, 2 * np.pi),
                (0, 1),
                (0, 2 * np.pi),
                (1, self.max_steps + 2),
            ]

            loss = []
            sols = []
            for i in range(rounds):
                start0 = np.random.uniform(-self.max_steps, self.max_steps + 0.5)
                period0 = np.random.uniform(1, self.max_steps + 2)
                x0 = np.array(
                    [
                        self.max_steps + 0.5,
                        np.pi,
                        2 * np.pi,
                        np.pi,
                        2 * np.pi,
                        1,
                        2 * np.pi,
                    ]
                ) * np.random.rand(7)
                x0 = np.hstack((start0, x0, period0))
                opt = minimize(
                    self.objective_harmonic,
                    x0,
                    args=(self.end_prob),
                    method="SLSQP",
                    bounds=bounds,
                    tol=1e-8,
                    options={"maxiter": 200000},
                )

                sols.append(opt.x)
                loss.append(self.objective_harmonic(opt.x, self.end_prob))

            best_sol = sols[np.argmin(loss)]

            start1, steps1, theta1, lamb1, theta2, lamb2, a1, phi1, period1 = best_sol

            U_sol1 = np.array(
                [
                    [np.cos(theta1 / 2), -np.exp(lamb1 * 1j) * np.sin(theta1 / 2)],
                    [np.sin(theta1 / 2), np.exp(lamb1 * 1j) * np.cos(theta1 / 2)],
                ]
            )
            U_sol2 = np.array(
                [
                    [np.cos(theta2 / 2), -np.exp(lamb2 * 1j) * np.sin(theta2 / 2)],
                    [np.sin(theta2 / 2), np.exp(lamb2 * 1j) * np.cos(theta2 / 2)],
                ]
            )

            init_amp = np.array(
                [
                    1 if i == int(start1) else 0
                    for i in range(-self.max_steps, self.max_steps + 1)
                ]
            )

            steps1 = min(int(steps1), self.max_steps - int(abs(start1)))

            coin_state1 = np.array([a1, (1 - a1**2) ** 0.5 * np.exp(1j * phi1)])

            fit_prob = self.simulator.simulate(
                coin_state1, [U_sol1, U_sol2], init_amp, steps1, int(period1)
            )

            n = max(
                len(np.where(self.end_prob != 0)[0]), len(np.where(fit_prob != 0)[0])
            )

            best_loss = np.min(loss) / n

            print(
                f"MSE: {best_loss}, std : {np.std(loss) / n}, x0: {int(start1)}, steps: {steps1}, θ1: {theta1:.3f}, λ1: {lamb1:.3f}, θ2: {theta2:.3f}, λ2: {lamb2:.3f}, a: {a1:.3f}, ϕ: {phi1:.3f}, period: {int(period1)}"
            )

            output = {
                "x0": int(start1),
                "steps": steps1,
                "θ1": theta1,
                "λ1": lamb1,
                "θ2": theta2,
                "λ2": lamb2,
                "a": a1,
                "ϕ": phi1,
                "period": int(period1),
            }

            if plot:
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_axes([0, 0, 1, 1])
                sns.set_style("darkgrid")
                plt.gca().set_facecolor("black")

                ax.bar(
                    range(-self.max_steps, self.max_steps + 1),
                    fit_prob,
                    label="fitter",
                    width=0.7 * (0.5 + 0.01 * self.max_steps),
                    align="edge",
                    color="rebeccapurple",
                )
                ax.bar(
                    range(-self.max_steps, self.max_steps + 1),
                    self.end_prob,
                    label="target",
                    width=-0.5 * (0.5 + 0.01 * self.max_steps),
                    align="edge",
                    color="magenta",
                )
                ax.bar(
                    [0],
                    [0],
                    label=f"MSE = {1000 * best_loss:.5f}e-3",
                    color="white",
                    alpha=0.01,
                )
                ax.set_xlabel("Positions", fontsize=15)
                ax.set_ylabel("Probability", fontsize=15)
                ax.set_title(
                    f"Harmonic Quantum Random Walks Fitter \n $x_{0}$={int(start1)},  Steps={steps1},  θ1={round(theta1, 3)},  λ1={round(lamb1, 3)},  a={round(a1, 3)},  ϕ={round(phi1, 3)}, θ2={round(theta2, 3)},  λ2={round(lamb2, 3)}, period={int(period1)}",
                    fontsize=15,
                )
                ax.legend(fontsize=15)
                plt.show()

        return output, best_loss, fit_prob
