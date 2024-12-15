#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================================================
Quantum-Enhanced Language Model (QELM) - Trainer with multi thread support. *CPU/GPU*
====================================================================================================

This script defines a Quantum-Enhanced Language Model (QELM) with the following features:
1. Gradient-Based Optimization using the Parameter Shift Rule.
2. Advanced Quantum Circuit Design with entangling gates and multiple layers.
3. Support for both Synthetic and Real Datasets resembling language data.
4. Enhanced Model Architecture with residual connections and layer normalization.
5. Robust Parameter Persistence with versioning and validation using a custom .qelm file extension.
6. User-Friendly Graphical User Interface (GUI) using Tkinter for training, inference, saving, loading, and exploring token mappings.
7. **Fixed Issues:**
   - Resolved shape mismatch error during residual connections by ensuring consistent tensor shapes across model components.
   - Added global exception handling to prevent the script from closing unexpectedly and to display error messages to the user.
   - Suppressed excessive logging from worker processes to prevent resource exhaustion.

Dependencies:
- qiskit
- qiskit-aer
- numpy
- scipy
- nltk
- tkinter

Ensure all dependencies are installed before running the script.

Check with Qiskit to ensure calls are correct. They have a tendency to change them with updates.

====================================================================================================

import sys
import os
import json
import time
import logging
import traceback
import threading
import multiprocessing
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter

try:
    import psutil
except ImportError:
    psutil = None

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Initialize NLTK quietly
nltk.download('punkt', quiet=True)

# Logging Configuration
logging.basicConfig(
    filename='qelm_enhanced.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have unit length.
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec.copy()
    return vec / norm


class QuantumParameterStore:
    """
    Stores quantum parameters with utilities for setting and retrieving values.
    """
    def __init__(self, size: int, prefix: str = "theta"):
        self.size = size
        self.parameters = [Parameter(f"{prefix}_{i}") for i in range(size)]
        self.values = np.zeros(size, dtype=float)

    def set_values(self, vals: np.ndarray):
        if vals.shape[0] != self.size:
            raise ValueError("Parameter size mismatch.")
        self.values = vals.copy()

    def get_values(self) -> np.ndarray:
        return self.values.copy()

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "prefix": self.parameters[0].name.rsplit('_', 1)[0],
            "values": self.values.tolist()
        }

    def from_dict(self, d: dict):
        if d["size"] != self.size:
            raise ValueError("Parameter size mismatch when loading parameters.")
        self.set_values(np.array(d["values"], dtype=float))


class QuantumLayerBase:
    """
    Base class for quantum layers that sets up simulators and provides circuit building utilities.
    """
    def __init__(self, sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True):
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.backend = self.initialize_simulator()

    def initialize_simulator(self):
        # Attempt GPU, fall back to CPU
        if self.sim_method == 'gpu':
            try:
                backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
                if self.enable_logging:
                    logging.info(f"{self.__class__.__name__}: Using GPU.")
            except Exception as e:
                if self.enable_logging:
                    logging.warning(f"{self.__class__.__name__} GPU init error: {e}, using CPU fallback.")
                backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using CPU.")
        return backend

    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        qubits_needed = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits_needed)

        # Prepare the state
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        state_prep_vec[:len(input_vector)] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, qubits=range(qubits_needed))

        # Simple parameterized ansatz
        num_layers = 2
        # Apply RY rotations and CX entanglements
        for layer in range(num_layers):
            for i in range(qubits_needed):
                theta = param_store.values[layer * qubits_needed + i]
                circuit.ry(theta, i)
            for i in range(qubits_needed - 1):
                circuit.cx(i, i + 1)

        # Final layer of rotations
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)

        circuit.save_statevector()
        return circuit

    def simulate(self, circuit: QuantumCircuit) -> np.ndarray:
        job = self.backend.run(circuit, shots=1)  # Statevector simulation doesn't need multiple shots
        result = job.result()
        final_state = result.get_statevector(circuit)
        return final_state.data


class QuantumAttentionLayer(QuantumLayerBase):
    """
    Quantum Attention Layer for the language model.
    Implements parameter sharing to reduce total parameters.
    """
    def __init__(self, embed_dim: int, num_heads: int, sim_method: str = 'cpu', num_threads: int = 1, prefix: str = "attn", enable_logging: bool = True):
        super().__init__(sim_method=sim_method, num_threads=num_threads, enable_logging=enable_logging)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        # Initialize parameter stores with parameter sharing across heads
        shared_size = (embed_dim * embed_dim) // num_heads
        self.query_params = QuantumParameterStore(shared_size, prefix=f"{prefix}_Q")
        self.key_params   = QuantumParameterStore(shared_size, prefix=f"{prefix}_K")
        self.value_params = QuantumParameterStore(shared_size, prefix=f"{prefix}_V")
        self.out_params   = QuantumParameterStore(shared_size, prefix=f"{prefix}_O")

    def forward(self, input_vector: np.ndarray, mode: str = 'query') -> np.ndarray:
        input_vector = normalize_vector(input_vector)
        if mode == 'query':
            param_store = self.query_params
        elif mode == 'key':
            param_store = self.key_params
        elif mode == 'value':
            param_store = self.value_params
        elif mode == 'out':
            param_store = self.out_params
        else:
            raise ValueError("Invalid mode for Attention forward.")

        circuit = self.build_circuit(input_vector, param_store)
        final_state = self.simulate(circuit)

        output_length = self.embed_dim
        if len(final_state) < output_length:
            output_vec = np.real(final_state[:len(final_state)])
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state[:output_length])

        return normalize_vector(output_vec)

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.query_params.get_values(),
            self.key_params.get_values(),
            self.value_params.get_values(),
            self.out_params.get_values()
        ])

    def set_all_parameters(self, params: np.ndarray):
        attn_size = (self.query_params.size + self.key_params.size +
                     self.value_params.size + self.out_params.size)
        if params.shape[0] != attn_size:
            raise ValueError("Param size mismatch in Attention.")
        q_size = self.query_params.size
        k_size = self.key_params.size
        v_size = self.value_params.size
        o_size = self.out_params.size

        self.query_params.set_values(params[:q_size])
        self.key_params.set_values(params[q_size:q_size+k_size])
        self.value_params.set_values(params[q_size+k_size:q_size+k_size+v_size])
        self.out_params.set_values(params[q_size+k_size+v_size:])

    def to_dict(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "query_params": self.query_params.to_dict(),
            "key_params": self.key_params.to_dict(),
            "value_params": self.value_params.to_dict(),
            "out_params": self.out_params.to_dict(),
            "sim_method": self.sim_method,
            "num_threads": self.num_threads
        }

    def from_dict(self, d: dict):
        if d["embed_dim"] != self.embed_dim or d["num_heads"] != self.num_heads:
            raise ValueError("Attention config mismatch.")
        self.query_params.from_dict(d["query_params"])
        self.key_params.from_dict(d["key_params"])
        self.value_params.from_dict(d["value_params"])
        self.out_params.from_dict(d["out_params"])
        self.sim_method = d.get("sim_method", "cpu")
        self.num_threads = d.get("num_threads", 1)
        self.backend = self.initialize_simulator()


class QuantumFeedForwardLayer(QuantumLayerBase):
    """
    Quantum Feed-Forward Layer for the language model.
    Implements parameter sharing to reduce total parameters.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, sim_method: str = 'cpu', num_threads: int = 1, prefix: str = "ffn", enable_logging: bool = True):
        super().__init__(sim_method=sim_method, num_threads=num_threads, enable_logging=enable_logging)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        shared_size = (embed_dim * hidden_dim) // 2  # Example: share parameters across two sub-layers
        self.w1_params = QuantumParameterStore(shared_size, prefix=f"{prefix}_W1")
        self.w2_params = QuantumParameterStore(shared_size, prefix=f"{prefix}_W2")

    def forward(self, input_vector: np.ndarray, layer: str = 'w1') -> np.ndarray:
        input_vector = normalize_vector(input_vector)
        if layer == 'w1':
            param_store = self.w1_params
            output_length = self.hidden_dim
        elif layer == 'w2':
            param_store = self.w2_params
            output_length = self.embed_dim
        else:
            raise ValueError("Invalid layer in FFN forward.")

        circuit = self.build_circuit(input_vector, param_store)
        final_state = self.simulate(circuit)

        if len(final_state) < output_length:
            output_vec = np.real(final_state[:len(final_state)])
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state[:output_length])

        return normalize_vector(output_vec)

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.w1_params.get_values(), self.w2_params.get_values()])

    def set_all_parameters(self, params: np.ndarray):
        ffn_size = self.w1_params.size + self.w2_params.size
        if params.shape[0] != ffn_size:
            raise ValueError("FFN param size mismatch.")
        w1_size = self.w1_params.size
        self.w1_params.set_values(params[:w1_size])
        self.w2_params.set_values(params[w1_size:])

    def to_dict(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "w1_params": self.w1_params.to_dict(),
            "w2_params": self.w2_params.to_dict(),
            "sim_method": self.sim_method,
            "num_threads": self.num_threads
        }

    def from_dict(self, d: dict):
        if d["embed_dim"] != self.embed_dim or d["hidden_dim"] != self.hidden_dim:
            raise ValueError("FFN config mismatch.")
        self.w1_params.from_dict(d["w1_params"])
        self.w2_params.from_dict(d["w2_params"])
        self.sim_method = d.get("sim_method", "cpu")
        self.num_threads = d.get("num_threads", 1)
        self.backend = self.initialize_simulator()


class AdamOptimizer:
    """
    Adam Optimizer for parameter updates.
    """
    def __init__(self, parameters: np.ndarray, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = np.zeros_like(self.parameters)
        self.v = np.zeros_like(self.parameters)
        self.t = 0

    def step(self, gradients: np.ndarray):
        self.t += 1
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * gradients
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (gradients ** 2)
        m_hat = self.m / (1 - self.betas[0] ** self.t)
        v_hat = self.v / (1 - self.betas[1] ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        self.parameters -= update
        return self.parameters


class QuantumLanguageModel:
    """
    The main Quantum Language Model integrating attention and feed-forward layers.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int,
                 sim_method: str = 'cpu', num_threads: int = 1, enable_logging: bool = True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Initialize embeddings
        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)

        # Initialize quantum layers with logging control
        self.attn = QuantumAttentionLayer(embed_dim, num_heads, sim_method=sim_method,
                                         num_threads=num_threads, prefix="layer1_attn", enable_logging=enable_logging)
        self.ffn = QuantumFeedForwardLayer(embed_dim, hidden_dim, sim_method=sim_method,
                                          num_threads=num_threads, prefix="layer1_ffn", enable_logging=enable_logging)

        # Initialize linear layers
        self.W_proj = (np.random.randn(embed_dim, hidden_dim) * 0.01).astype(np.float32)  # Shape: (256, 512)
        self.W_out = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)  # Shape: (10000, 256)

        # Initialize quantum parameters
        self._initialize_quantum_params()

    def _initialize_quantum_params(self):
        scale = 0.1
        self.attn.query_params.set_values(np.random.randn(self.attn.query_params.size) * scale)
        self.attn.key_params.set_values(np.random.randn(self.attn.key_params.size) * scale)
        self.attn.value_params.set_values(np.random.randn(self.attn.value_params.size) * scale)
        self.attn.out_params.set_values(np.random.randn(self.attn.out_params.size) * scale)
        self.ffn.w1_params.set_values(np.random.randn(self.ffn.w1_params.size) * scale)
        self.ffn.w2_params.set_values(np.random.randn(self.ffn.w2_params.size) * scale)

    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        """
        Forward pass through the model.
        """
        if not input_ids:
            raise ValueError("input_ids is empty.")
        for idx in input_ids:
            if idx < 0 or idx >= self.vocab_size:
                raise ValueError(f"Input id {idx} out of range.")

        # Starting with the last input token
        x = self.embeddings[input_ids[-1]]

        # Attention
        attn_output_query = self.attn.forward(x, mode='query')  # (256,)
        attn_output_key = self.attn.forward(x, mode='key')      # (256,)
        attn_output_value = self.attn.forward(x, mode='value')  # (256,)
        attn_output_out = self.attn.forward(x, mode='out')      # (256,)

        # Combine attention outputs
        attn_output = attn_output_query + attn_output_key + attn_output_value + attn_output_out  # (256,)

        if use_residual:
            x = normalize_vector(x + attn_output)
        else:
            x = attn_output

        # Feed Forward
        ffn_output_w1 = self.ffn.forward(x, layer='w1')  # (512,)
        ffn_output_w2 = self.ffn.forward(ffn_output_w1, layer='w2')  # (256,)

        if use_residual:
            x = normalize_vector(x + ffn_output_w2)  # (256,) + (256,) = (256,)
        else:
            x = ffn_output_w2  # (256,)

        logits = self.W_out @ x  # (10000,256) @ (256,) = (10000,)

        return logits

    def get_all_parameters(self) -> np.ndarray:
        """
        Retrieve all parameters as a single numpy array.
        """
        return np.concatenate([
            self.attn.get_all_parameters(),
            self.ffn.get_all_parameters(),
            self.W_proj.flatten(),
            self.W_out.flatten()
        ])

    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single numpy array.
        """
        attn_size = (self.attn.query_params.size + self.attn.key_params.size +
                     self.attn.value_params.size + self.attn.out_params.size)
        ffn_size = self.ffn.w1_params.size + self.ffn.w2_params.size
        proj_size = self.embed_dim * self.hidden_dim  # (256 * 512) = 131072
        out_size = self.vocab_size * self.embed_dim  # (10000 * 256) = 2560000
        expected = attn_size + ffn_size + proj_size + out_size

        if params.shape[0] != expected:
            raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")

        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:attn_size+ffn_size])
        self.W_proj = params[attn_size+ffn_size:attn_size+ffn_size+proj_size].reshape(self.embed_dim, self.hidden_dim)
        self.W_out = params[attn_size+ffn_size+proj_size:].reshape(self.vocab_size, self.embed_dim)

    def to_dict(self) -> dict:
        """
        Serialize the model to a dictionary.
        """
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings.tolist(),
            "attn": self.attn.to_dict(),
            "ffn": self.ffn.to_dict(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "version": "4.0"  # Updated version
        }

    def from_dict(self, d: dict):
        """
        Deserialize the model from a dictionary.
        """
        if (d["vocab_size"] != self.vocab_size or
            d["embed_dim"] != self.embed_dim or
            d["num_heads"] != self.num_heads or
            d["hidden_dim"] != self.hidden_dim):
            raise ValueError("Model config mismatch.")

        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)

    def save_model(self, save_path: str):
        """
        Save the model to a JSON file.
        """
        model_dict = self.to_dict()
        with open(save_path, 'w') as f:
            json.dump(model_dict, f)
        logging.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load the model from a JSON file.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")
        with open(load_path, 'r') as f:
            model_dict = json.load(f)
        if "version" not in model_dict or model_dict["version"] != "4.0":
            raise ValueError("Unsupported model version.")
        self.from_dict(model_dict)
        logging.info(f"Model loaded from {load_path}")

    def shift_parameter(self, param_index: int, shift: float):
        """
        Shift a specific parameter by a given amount.
        """
        shifted_params = self.get_all_parameters()
        shifted_params[param_index] += shift
        self.set_all_parameters(shifted_params)

    def unshift_parameter(self, param_index: int, shift: float):
        """
        Restore a specific parameter by unshifting it.
        """
        self.shift_parameter(param_index, -shift)


def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset for training.
    """
    X = np.random.randint(4, vocab_size, size=(num_samples,))  # Start from index 4 to reserve special tokens
    Y = np.random.randint(4, vocab_size, size=(num_samples,))
    return X, Y


def load_real_dataset(file_path: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load and preprocess a real dataset from a text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1

    # Add special tokens
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # Reserve indices for special tokens
    token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in sorted_tokens[:vocab_size - len(special_tokens)]:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    X, Y_ids = [], []
    for i in range(len(tokens)-1):
        current_token = tokens[i]
        next_token = tokens[i+1]
        if current_token in token_to_id and next_token in token_to_id:
            X.append(token_to_id[current_token])
            Y_ids.append(token_to_id[next_token])
        else:
            X.append(token_to_id.get(current_token, token_to_id["<UNK>"]))
            Y_ids.append(token_to_id.get(next_token, token_to_id["<UNK>"]))

    Y = np.array(Y_ids, dtype=np.int32)

    return np.array(X), Y, token_to_id


def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    """
    Cross-Entropy Loss for next-token prediction.
    """
    # Numerical stability
    logits = logits - np.max(logits)
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    # Avoid log(0)
    softmax = np.clip(softmax, 1e-12, 1.0)
    return -np.log(softmax[target])


def perplexity(logits: np.ndarray, target: int) -> float:
    """
    Calculate Perplexity for a single prediction.
    """
    ce_loss = cross_entropy_loss(logits, target)
    return np.exp(ce_loss)


def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score between reference and hypothesis.
    Simplistic implementation for demonstration purposes.
    """
    from collections import Counter
    import math

    weights = [1.0 / max_n] * max_n
    reference_counts = [Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]) for n in range(1, max_n+1)]
    hypothesis_counts = [Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]) for n in range(1, max_n+1)]

    precisions = []
    for ref_count, hyp_count in zip(reference_counts, hypothesis_counts):
        overlap = hyp_count & ref_count
        precision = sum(overlap.values()) / max(sum(hyp_count.values()), 1e-12)
        precisions.append(precision)

    # Brevity penalty
    ref_length = len(reference)
    hyp_length = len(hypothesis)
    if hyp_length == 0:
        bp = 0
    elif hyp_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / hyp_length)

    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
    else:
        geo_mean = 0

    bleu = bp * geo_mean
    return bleu


def compute_gradient_for_parameter(args):
    """
    Compute gradient for a single parameter using the parameter-shift rule with Cross-Entropy Loss.
    Intended for parallel execution.
    """
    (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads, X, Y, original_params, i) = args
    try:
        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            sim_method=sim_method,
            num_threads=num_threads,
            enable_logging=False  # Suppress logging during gradient computation
        )
        model.set_all_parameters(original_params)

        # Parameter shift
        shift = np.pi / 2
        model.shift_parameter(i, shift)
        loss_plus = np.mean([
            cross_entropy_loss(model.forward([x]), y)
            for x, y in zip(X, Y)
        ])

        model.unshift_parameter(i, shift)
        model.shift_parameter(i, -shift)
        loss_minus = np.mean([
            cross_entropy_loss(model.forward([x]), y)
            for x, y in zip(X, Y)
        ])

        model.unshift_parameter(i, -shift)

        gradient = (loss_plus - loss_minus) / 2.0
        return i, gradient
    except Exception:
        traceback.print_exc()
        return i, 0.0


def compute_gradients_parallel(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, num_processes: int = 1) -> np.ndarray:
    """
    Compute gradients for all parameters in parallel.
    """
    gradients = np.zeros_like(model.get_all_parameters())
    original_params = model.get_all_parameters().copy()
    total_params = len(original_params)

    args_list = [
        (
            model.vocab_size,
            model.embed_dim,
            model.num_heads,
            model.hidden_dim,
            model.attn.sim_method,
            model.attn.num_threads,
            X,
            Y,
            original_params,
            i
        )
        for i in range(total_params)
    ]

    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(compute_gradient_for_parameter, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            i, gradient = future.result()
            gradients[i] = gradient

    return gradients


def train_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray,
                epochs: int = 10, lr: float = 0.001, num_threads: int = 1,
                log_callback=None, stop_flag=None, time_lock: threading.Lock = None, time_data=None,
                optimizer=None):
    """
    Train the Quantum Language Model using the provided optimizer.
    """
    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_callback:
                log_callback("Training stopped by user.\n")
            break

        if log_callback:
            log_callback(f"Starting Epoch {epoch+1}/{epochs}\n")

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads)

        # Update parameters using optimizer
        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            # Fallback to simple gradient descent
            params = model.get_all_parameters()
            params -= lr * gradients
            model.set_all_parameters(params)

        # Compute average loss and perplexity
        total_loss = np.mean([
            cross_entropy_loss(model.forward([x]), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward([x]), y)
            for x, y in zip(X, Y)
        ])

        if log_callback:
            log_callback(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        # Update evaluation metrics (BLEU score can be computed on a validation set if available)

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = time.time() - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

    if log_callback and (not stop_flag or not stop_flag.is_set()):
        log_callback("Training completed.\n")


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):
    """
    Generate a sequence of tokens based on the input sequence.
    """
    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward([generated[-1]])
        probabilities = softmax(logits / temperature)

        # Sample from the probability distribution
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)

        # Stop generation if <END> token is generated
        if chosen_index == token_to_id.get("<END>", chosen_index):
            break

    # Convert token IDs back to tokens
    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)

    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")

    return generated_tokens, response


class QELM_GUI:
    """
    Graphical User Interface for the Quantum-Enhanced Language Model.
    """
    def __init__(self, master):
        self.master = master
        master.title("QELM Trainer - Enhanced")
        master.geometry("1400x900")  # Adjusted window size for better layout
        master.resizable(False, False)

        # Model parameters
        self.vocab_size = 10000  # Increased vocab size
        self.embed_dim = 256      # Increased embedding dimension
        self.num_heads = 8        # Increased number of attention heads
        self.hidden_dim = 512     # Increased hidden dimension
        self.sim_method = 'cpu'
        self.num_threads = min(8, multiprocessing.cpu_count())  # Adjusted for higher model complexity
        self.model = QuantumLanguageModel(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim,
                                          sim_method=self.sim_method, num_threads=self.num_threads, enable_logging=True)

        self.token_to_id = {}
        self.id_to_token = {}

        # Initialize optimizer
        self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=0.001)

        # Training controls
        self.stop_flag = threading.Event()
        self.time_data = {'start_time': 0, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
        self.time_lock = threading.Lock()

        # Initialize per-process CPU usage monitoring
        if psutil:
            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)  # Initialize
        else:
            self.process = None

        # Configure GUI appearance
        self.master.configure(bg="#2C3E50")
        style = ttk.Style(self.master)
        style.theme_use('clam')
        style.configure(".", background="#2C3E50", foreground="white")
        style.configure("TFrame", background="#2C3E50")
        style.configure("TLabelFrame", background="#34495E", foreground="white")
        style.configure("TLabel", background="#2C3E50", foreground="white")
        style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
        style.configure("TNotebook", background="#2C3E50")
        style.configure("TNotebook.Tab", background="#34495E", foreground="white")
        style.configure("Horizontal.TProgressbar", background="#1ABC9C", troughcolor="#34495E")
        # Lighter background for Entry/Spinbox
        style.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
        style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
        style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])

        self.create_widgets()
        self.update_resource_usage()
        self.update_time_label()

    def create_widgets(self):
        """
        Create and arrange all widgets in the GUI.
        """
        container = ttk.Frame(self.master)
        container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(container)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)

        self.notebook = ttk.Notebook(left_frame)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        self.notebook.pack(fill='both', expand=True)

        # =======================
        # Train Model Tab
        # =======================
        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)

        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        select_dataset_btn = ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset)
        select_dataset_btn.pack(side='right', padx=10, pady=10)

        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Hyperparameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(hyperparams_frame, text="Epochs:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.epochs_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        sim_settings_frame = ttk.LabelFrame(self.tab_train, text="Simulation Settings")
        sim_settings_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(sim_settings_frame, text="Simulation Method:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.sim_method_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(sim_settings_frame, text='CPU', variable=self.sim_method_var, value='cpu', command=self.update_threads_based_on_method)
        gpu_radio = ttk.Radiobutton(sim_settings_frame, text='GPU', variable=self.sim_method_var, value='gpu', command=self.update_threads_based_on_method)
        cpu_radio.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        gpu_radio.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        ttk.Label(sim_settings_frame, text="Number of Threads:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(
            sim_settings_frame,
            from_=1,
            to=multiprocessing.cpu_count(),
            textvariable=self.num_threads_var,
            width=5
        )
        self.num_threads_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=1, column=2, padx=10, pady=10, sticky='w')

        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)

        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)

        stop_button = ttk.Button(train_controls_frame, text="STOP (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=10, pady=10)

        hard_stop_button = ttk.Button(train_controls_frame, text="HARD STOP (Immediate)", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=10, pady=10)

        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)

        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)

        self.progress = ttk.Progressbar(self.tab_train, orient='horizontal', length=600, mode='determinate')
        self.progress.pack(pady=10)

        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)

        # Evaluation Metrics within Train Model Tab
        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Evaluation Metrics")
        eval_metrics_frame.pack(fill='x', padx=10, pady=10)

        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.pack(anchor='w', padx=10, pady=5)

        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU Score: N/A")
        self.bleu_label.pack(anchor='w', padx=10, pady=5)

        # =======================
        # Run Inference Tab
        # =======================
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Max Length:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.max_length_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.max_length_entry.insert(0, "50")
        self.max_length_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Temperature:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.temperature_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)

        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)

        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)

        # =======================
        # Manage Token Mappings Tab
        # =======================
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)

        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)

        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                          bg="#2C3E50", fg="white", insertbackground="white")
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)

        # =======================
        # System Resources & Time
        # =======================
        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)

        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)

        self.gpu_label = ttk.Label(usage_frame, text="GPU: Check externally (e.g., nvidia-smi)")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)

        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating...")
        self.time_label.pack(anchor='w', padx=10, pady=5)

    def update_threads_based_on_method(self):
        """
        Update the maximum number of threads based on the simulation method.
        """
        method = self.sim_method_var.get()
        max_threads = multiprocessing.cpu_count()
        self.num_threads_spinbox.config(to=max_threads)
        if self.num_threads_var.get() > max_threads:
            self.num_threads_var.set(max_threads)

    def log_train(self, message: str):
        """
        Log messages to the training log.
        """
        self.train_log.config(state='normal')
        self.train_log.insert(tk.END, message)
        self.train_log.see(tk.END)
        self.train_log.config(state='disabled')

    def log_infer(self, message: str):
        """
        Log messages to the inference log.
        """
        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')

    def log_token_map(self, message: str):
        """
        Log messages to the token map display.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')

    def select_dataset(self):
        """
        Open a file dialog to select a dataset.
        """
        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File",
                                                   filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                self.dataset_path = file_path
                self.dataset_path_var.set(file_path)
                self.log_train(f"Selected Dataset: {file_path}\n")
                self.token_to_id = {}
                self.id_to_token = {}
        except Exception as e:
            err_msg = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Dataset Selection Error", err_msg)

    def train_model(self):
        """
        Start the training process in a separate thread.
        """
        try:
            epochs = int(self.epochs_entry.get())
            lr = float(self.lr_entry.get())
            if epochs <= 0 or lr <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for epochs and learning rate.")
            return

        sim_method = self.sim_method_var.get()
        num_threads = self.num_threads_var.get()
        max_threads = multiprocessing.cpu_count()
        if num_threads > max_threads:
            messagebox.showwarning("Thread Limit", f"Resetting threads to max {max_threads}")
            num_threads = max_threads
            self.num_threads_var.set(num_threads)

        # Load dataset
        if hasattr(self, 'dataset_path') and hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                X, Y, token_to_id = load_real_dataset(dataset_path, self.vocab_size)
                self.X = X
                self.Y = Y
                self.token_to_id = token_to_id
                self.id_to_token = {idx: token for token, idx in token_to_id.items()}
                self.log_train(f"Loaded real dataset from {dataset_path}\n")
            except Exception as e:
                err_msg = f"Failed to load dataset:\n{traceback.format_exc()}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("Dataset Load Error", err_msg)
                return
        else:
            X, Y = create_synthetic_dataset(self.vocab_size, num_samples=500)  # Increased samples for better training
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")

        # Update model simulation settings
        self.model.sim_method = sim_method
        self.model.num_threads = num_threads
        self.model.attn.sim_method = sim_method
        self.model.attn.num_threads = num_threads
        self.model.attn.backend = self.model.attn.initialize_simulator()

        self.model.ffn.sim_method = sim_method
        self.model.ffn.num_threads = num_threads
        self.model.ffn.backend = self.model.ffn.initialize_simulator()

        # Initialize optimizer with current parameters
        self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=lr)

        # Disable buttons during training
        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.infer_button.config(state='disabled')
        self.stop_flag.clear()

        self.progress['value'] = 0
        self.log_train("Starting training...\n")

        # Initialize time data
        with self.time_lock:
            self.time_data['start_time'] = time.time()
            self.time_data['epochs_done'] = 0
            self.time_data['epochs'] = epochs
            self.time_data['remaining'] = 0

        # Start training in a separate thread
        training_thread = threading.Thread(target=self.training_process, args=(epochs, num_threads))
        training_thread.start()

    def training_process(self, epochs: int, num_threads: int):
        """
        The actual training process running in a separate thread.
        """
        try:
            def log_callback(msg):
                self.log_train(msg)
                if "Epoch" in msg and "/" in msg:
                    # Update progress bar
                    parts = msg.split()
                    for p in parts:
                        if "/" in p and "Epoch" not in p:
                            try:
                                current, total = p.split("/")
                                current = int(current)
                                total = int(total)
                                percentage = (current / total) * 100
                                self.update_progress(percentage)
                            except:
                                pass

            train_model(
                self.model,
                self.X,
                self.Y,
                epochs=epochs,
                lr=self.optimizer.lr,
                num_threads=num_threads,
                log_callback=log_callback,
                stop_flag=self.stop_flag,
                time_lock=self.time_lock,
                time_data=self.time_data,
                optimizer=self.optimizer
            )
            if not self.stop_flag.is_set():
                self.log_train("Training completed successfully.\n")
                messagebox.showinfo("Training Completed", "Model training completed successfully.")
        except Exception as e:
            err_msg = f"Training error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Training Error", err_msg)
        finally:
            # Re-enable buttons after training
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.infer_button.config(state='normal')
            if not self.stop_flag.is_set():
                self.progress['value'] = 100

            # Update evaluation metrics
            self.evaluate_model()

    def stop_training(self):
        """
        Gracefully stop the training after the current epoch.
        """
        self.stop_flag.set()
        self.log_train("Stop signal sent. Will stop after current epoch.\n")

    def hard_stop(self):
        """
        Immediately terminate the application.
        """
        self.log_train("Hard stop invoked. Terminating immediately.\n")
        os._exit(1)

    def save_model(self):
        """
        Save the trained model and token mappings.
        """
        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm",
                                                     filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if save_path:
                self.model.save_model(save_path)
                if self.token_to_id:
                    token_map_path = save_path.replace(".qelm", "_token_map.json")
                    with open(token_map_path, 'w') as f:
                        json.dump(self.token_to_id, f, indent=4)
                    self.log_train(f"Token mappings saved to {token_map_path}\n")
                messagebox.showinfo("Model Saved", f"Model saved to {save_path}")
        except Exception as e:
            err_msg = f"Save model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Save Error", err_msg)

    def load_model(self):
        """
        Load a saved model and its token mappings.
        """
        try:
            load_path = filedialog.askopenfilename(title="Load Model",
                                                  filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if load_path:
                self.model.load_model(load_path)
                token_map_path = load_path.replace(".qelm", "_token_map.json")
                try:
                    with open(token_map_path, 'r') as f:
                        self.token_to_id = json.load(f)
                    self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                    self.log_train(f"Loaded token mappings from {token_map_path}\n")
                    self.display_token_map()
                except FileNotFoundError:
                    self.log_train("No token mappings file found.\n")
                messagebox.showinfo("Model Loaded", f"Model loaded from {load_path}")
        except Exception as e:
            err_msg = f"Load model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def run_inference(self):
        """
        Run inference based on the input token.
        """
        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return

        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for max length and temperature.")
            return

        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}' with max_length={max_length} and temperature={temperature}...\n")

        # Start inference in a separate thread
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature))
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):
        """
        The actual inference process running in a separate thread.
        """
        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Input token '{input_token}' not found in token mappings.")
            input_id = self.token_to_id[input_token]
            generated_tokens, response = run_inference(
                self.model,
                [input_id],
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=self.log_infer
            )
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')

    def load_token_map(self):
        """
        Load token-to-ID mappings from a JSON file.
        """
        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):
        """
        Display the token-to-ID mappings in the GUI.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_progress(self, percentage):
        """
        Update the training progress bar.
        """
        self.progress['value'] = percentage
        self.master.update_idletasks()

    def update_resource_usage(self):
        """
        Update CPU usage and display GPU status.
        """
        if self.process:
            cpu_usage = f"{self.process.cpu_percent(interval=None)}%"
        else:
            cpu_usage = "psutil not installed"

        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")

        self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
        """
        Update the elapsed and remaining training time.
        """
        with self.time_lock:
            elapsed = time.time() - self.time_data['start_time'] if self.time_data.get('start_time', 0) > 0 else 0
            elapsed_str = f"{elapsed:.1f}s"

            if self.time_data.get('epochs_done', 0) == 0 and self.time_data.get('epochs', 0) > 0:
                remaining_str = "Estimating..."
            else:
                remaining = self.time_data.get('remaining', 0)
                if remaining > 0:
                    remaining_str = f"{remaining:.1f}s"
                else:
                    if 0 < self.time_data.get('epochs_done', 0) < self.time_data.get('epochs', 0):
                        remaining_str = "Estimating..."
                    else:
                        remaining_str = "0s"

        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

        # Update every second
        self.master.after(1000, self.update_time_label)

    def evaluate_model(self):
        """
        Evaluate the model using Perplexity and BLEU score.
        """
        # Compute Perplexity on training data
        perplexities = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x])
            perp = perplexity(logits, y)
            perplexities.append(perp)
        avg_perplexity = np.mean(perplexities)

        # Compute BLEU score (requires reference and hypothesis; simplistic implementation)
        # Here, we treat training data as reference and model's predictions as hypothesis
        # This is not standard but serves demonstration purposes
        hypotheses = []
        references = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x])
            predicted = np.argmax(logits)
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(y, "<UNK>")])

        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            bleu = bleu_score(ref, hyp)
            bleu_scores.append(bleu)
        avg_bleu = np.mean(bleu_scores)

        # Update GUI labels
        self.perplexity_label.config(text=f"Perplexity: {avg_perplexity:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")


def main():
    """
    Entry point for the application.
    """
    try:
        root = tk.Tk()
        gui = QELM_GUI(root)
        root.mainloop()
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.critical(f"Unexpected error:\n{error_trace}")
        hidden_root = tk.Tk()
        hidden_root.withdraw()
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
