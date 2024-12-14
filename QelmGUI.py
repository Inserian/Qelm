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
"""

import sys
import os
import json
import time
import logging
import traceback
import threading
import multiprocessing
import concurrent.futures
from typing import List, Dict
from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import nltk
from nltk.tokenize import word_tokenize
from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk
from tkinter import ttk
from multiprocessing import freeze_support

try:
    import psutil
except ImportError:
    psutil = None

# Initialize NLTK data (only once)
nltk.download('punkt', quiet=True)

# Logging configuration
logging.basicConfig(
    filename='qelm.log',
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
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


class QuantumParameterStore:
    def __init__(self, size: int, prefix: str = "theta"):
        self.size = size
        self.parameters = [Parameter(f"{prefix}_{i}") for i in range(size)]
        self.values = np.zeros(size, dtype=float)
    
    def set_values(self, vals: np.ndarray):
        if vals.shape[0] != self.size:
            sys.exit("Error: Parameter values length mismatch.")
        self.values = vals
    
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
            sys.exit("Error: Size mismatch when loading parameters.")
        self.set_values(np.array(d["values"], dtype=float))


class QuantumAttentionLayer:
    def __init__(self, embed_dim: int, num_heads: int, sim_method: str = 'cpu', num_threads: int = 1, prefix: str = "attn"):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            sys.exit("Error: embed_dim must be divisible by num_heads.")
        
        self.query_params = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_Q")
        self.key_params   = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_K")
        self.value_params = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_V")
        self.out_params   = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_O")
        
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.backend = self.initialize_simulator()
    
    def initialize_simulator(self):
        if self.sim_method == 'gpu':
            try:
                backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
                logging.info("Attention Layer using GPU.")
            except Exception as e:
                logging.error(f"Attention GPU init error: {e}, falling back to CPU.")
                backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            logging.info("Attention Layer using CPU.")
        return backend
    
    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        qubits_needed = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits_needed)
        
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        state_prep_vec[:len(input_vector)] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, qubits=range(qubits_needed))
        
        num_layers = 2
        for layer in range(num_layers):
            for i in range(qubits_needed):
                theta = param_store.values[layer * qubits_needed + i]
                circuit.ry(theta, i)
            for i in range(qubits_needed - 1):
                circuit.cx(i, i+1)
        
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)
        
        circuit.save_statevector()
        return circuit
    
    def forward(self, input_vector: np.ndarray, mode: str = 'query') -> np.ndarray:
        input_vector = normalize_vector(input_vector)
        if mode not in ['query', 'key', 'value', 'out']:
            sys.exit("Invalid mode in Attention forward")
        
        if mode == 'query':
            param_store = self.query_params
        elif mode == 'key':
            param_store = self.key_params
        elif mode == 'value':
            param_store = self.value_params
        else:
            param_store = self.out_params

        circuit = self.build_circuit(input_vector, param_store)
        
        try:
            job = self.backend.run(circuit, shots=1024)
            result = job.result()
            final_state = result.get_statevector(circuit)
        except Exception as e:
            logging.error(f"Quantum simulation error in Attention ({mode}): {e}")
            sys.exit(1)
        
        output_length = self.embed_dim
        if len(final_state.data) < output_length:
            logging.warning("State vector shorter than embed_dim. Padding.")
            output_vec = np.real(final_state.data[:len(final_state.data)])
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state.data[:output_length])
        
        output_vec = normalize_vector(output_vec)
        logging.info(f"Attention {mode} forward done. Output shape {output_vec.shape}")
        return output_vec
    
    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.query_params.get_values(),
            self.key_params.get_values(),
            self.value_params.get_values(),
            self.out_params.get_values()
        ])
    
    def set_all_parameters(self, params: np.ndarray):
        total_size = (self.query_params.size + self.key_params.size +
                      self.value_params.size + self.out_params.size)
        if params.shape[0] != total_size:
            sys.exit("Param size mismatch in Attention.")
        
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
            sys.exit("Attention config mismatch.")
        self.query_params.from_dict(d["query_params"])
        self.key_params.from_dict(d["key_params"])
        self.value_params.from_dict(d["value_params"])
        self.out_params.from_dict(d["out_params"])
        self.sim_method = d.get("sim_method", "cpu")
        self.num_threads = d.get("num_threads", 1)
        self.backend = self.initialize_simulator()


class QuantumFeedForwardLayer:
    def __init__(self, embed_dim: int, hidden_dim: int, sim_method: str = 'cpu', num_threads: int = 1, prefix: str = "ffn"):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.w1_params = QuantumParameterStore(embed_dim * hidden_dim, prefix=f"{prefix}_W1")
        self.w2_params = QuantumParameterStore(hidden_dim * embed_dim, prefix=f"{prefix}_W2")
        
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.backend = self.initialize_simulator()
    
    def initialize_simulator(self):
        if self.sim_method == 'gpu':
            try:
                backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
                logging.info("FFN Layer using GPU.")
            except Exception as e:
                logging.error(f"FFN GPU init error: {e}, falling back to CPU.")
                backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            logging.info("FFN Layer using CPU.")
        return backend
    
    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        qubits_needed = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits_needed)
        
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        state_prep_vec[:len(input_vector)] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, range(qubits_needed))
        
        num_layers = 2
        for layer in range(num_layers):
            for i in range(qubits_needed):
                theta = param_store.values[layer * qubits_needed + i]
                circuit.ry(theta, i)
            for i in range(qubits_needed - 1):
                circuit.cx(i, i+1)
        
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)
        
        circuit.save_statevector()
        return circuit
    
    def forward(self, input_vector: np.ndarray, layer: str = 'w1') -> np.ndarray:
        input_vector = normalize_vector(input_vector)
        if layer not in ['w1', 'w2']:
            sys.exit("Invalid layer in FFN forward.")
        
        param_store = self.w1_params if layer == 'w1' else self.w2_params
        circuit = self.build_circuit(input_vector, param_store)
        
        try:
            job = self.backend.run(circuit, shots=1024)
            result = job.result()
            final_state = result.get_statevector(circuit)
        except Exception as e:
            logging.error(f"FFN simulation error in {layer}: {e}")
            sys.exit(1)
        
        output_length = self.hidden_dim
        if len(final_state.data) < output_length:
            logging.warning("FFN state vector shorter than hidden_dim. Padding.")
            output_vec = np.real(final_state.data[:len(final_state.data)])
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state.data[:output_length])
        
        output_vec = normalize_vector(output_vec)
        logging.info(f"FFN {layer} forward done. Output shape {output_vec.shape}")
        return output_vec
    
    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.w1_params.get_values(), self.w2_params.get_values()])
    
    def set_all_parameters(self, params: np.ndarray):
        total_size = self.w1_params.size + self.w2_params.size
        if params.shape[0] != total_size:
            sys.exit("FFN param size mismatch.")
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
            sys.exit("FFN config mismatch.")
        self.w1_params.from_dict(d["w1_params"])
        self.w2_params.from_dict(d["w2_params"])
        self.sim_method = d.get("sim_method", "cpu")
        self.num_threads = d.get("num_threads", 1)
        self.backend = self.initialize_simulator()


class QuantumLanguageModel:
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int, sim_method: str = 'cpu', num_threads: int = 1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        self.embeddings = (np.random.randn(vocab_size, embed_dim)*0.01).astype(np.float32)
        
        self.attn = QuantumAttentionLayer(embed_dim, num_heads, sim_method=sim_method, num_threads=num_threads, prefix="layer1_attn")
        self.ffn  = QuantumFeedForwardLayer(embed_dim, hidden_dim, sim_method=sim_method, num_threads=num_threads, prefix="layer1_ffn")
        
        self.W_proj = np.random.randn(embed_dim, hidden_dim).astype(np.float32)*0.01
        self.W_out = np.random.randn(vocab_size, embed_dim).astype(np.float32)*0.01
        
        self._initialize_quantum_params()
    
    def _initialize_quantum_params(self):
        scale = 0.1
        self.attn.query_params.set_values(np.random.randn(self.attn.query_params.size)*scale)
        self.attn.key_params.set_values(np.random.randn(self.attn.key_params.size)*scale)
        self.attn.value_params.set_values(np.random.randn(self.attn.value_params.size)*scale)
        self.attn.out_params.set_values(np.random.randn(self.attn.out_params.size)*scale)
        self.ffn.w1_params.set_values(np.random.randn(self.ffn.w1_params.size)*scale)
        self.ffn.w2_params.set_values(np.random.randn(self.ffn.w2_params.size)*scale)
    
    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        if not input_ids:
            sys.exit("Error: input_ids is empty.")
        
        try:
            x = self.embeddings[input_ids[0]]
        except IndexError:
            sys.exit(f"Error: input_id {input_ids[0]} out of vocab range {self.vocab_size}.")
        
        attn_output = self.attn.forward(x, mode='query')
        key_output = self.attn.forward(x, mode='key')
        value_output = self.attn.forward(x, mode='value')
        
        combined_attn = attn_output + key_output + value_output
        
        if use_residual:
            if x.shape[0] != combined_attn.shape[0]:
                sys.exit("Shape mismatch in attn residual.")
            x = normalize_vector(x + combined_attn)
        else:
            x = combined_attn
        
        ffn_output_w1 = self.ffn.forward(x, layer='w1')
        ffn_output_w2 = self.ffn.forward(ffn_output_w1, layer='w2')
        ffn_output = ffn_output_w1 + ffn_output_w2
        
        ffn_output_proj = self.W_proj @ ffn_output
        
        if use_residual:
            if x.shape[0] != ffn_output_proj.shape[0]:
                sys.exit("Shape mismatch in ffn residual.")
            x = normalize_vector(x + ffn_output_proj)
        else:
            x = ffn_output
        
        logits = self.W_out @ x
        return logits
    
    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.attn.get_all_parameters(),
            self.ffn.get_all_parameters(),
            self.W_proj.flatten(),
            self.W_out.flatten()
        ])
    
    def set_all_parameters(self, params: np.ndarray):
        attn_size = self.attn.query_params.size + self.attn.key_params.size + self.attn.value_params.size + self.attn.out_params.size
        ffn_size = self.ffn.w1_params.size + self.ffn.w2_params.size
        proj_size = self.embed_dim * self.hidden_dim
        out_size = self.vocab_size * self.embed_dim
        expected = attn_size + ffn_size + proj_size + out_size
        if params.shape[0] != expected:
            sys.exit(f"Param mismatch. Expected {expected}, got {params.shape[0]}.")
        
        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:attn_size+ffn_size])
        self.W_proj = params[attn_size+ffn_size:attn_size+ffn_size+proj_size].reshape(self.embed_dim, self.hidden_dim)
        self.W_out = params[attn_size+ffn_size+proj_size:].reshape(self.vocab_size, self.embed_dim)
    
    def to_dict(self) -> dict:
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
            "version": "1.0"
        }
    
    def from_dict(self, d: dict):
        if (d["vocab_size"] != self.vocab_size or 
            d["embed_dim"] != self.embed_dim or
            d["num_heads"] != self.num_heads or
            d["hidden_dim"] != self.hidden_dim):
            sys.exit("Model config mismatch.")
        
        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)
    
    def save_model(self, save_path: str):
        model_dict = self.to_dict()
        try:
            with open(save_path, 'w') as f:
                json.dump(model_dict, f)
            logging.info(f"Model saved: {save_path}")
        except Exception as e:
            logging.error(f"Save model error: {e}")
            sys.exit(1)
    
    def load_model(self, load_path: str):
        try:
            with open(load_path, 'r') as f:
                model_dict = json.load(f)
        except FileNotFoundError:
            sys.exit(f"File {load_path} does not exist.")
        except json.JSONDecodeError:
            sys.exit("Invalid JSON in model file.")
        except Exception as e:
            sys.exit(f"Error reading model file: {e}")
        
        if "version" not in model_dict or model_dict["version"] != "1.0":
            sys.exit("Unsupported model version.")
        
        try:
            self.from_dict(model_dict)
            logging.info(f"Model loaded from {load_path}")
        except Exception as e:
            logging.error(f"Load model error: {e}")
            sys.exit(1)


def create_synthetic_dataset(vocab_size: int, num_samples: int = 100):
    X = np.random.randint(0, vocab_size, size=(num_samples,))
    Y = np.zeros((num_samples, vocab_size), dtype=np.float32)
    for i in range(num_samples):
        target_id = np.random.randint(0, vocab_size)
        Y[i, target_id] = 1.0
    return X, Y


def load_real_dataset(file_path: str, vocab_size: int):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"File {file_path} does not exist.")
    except Exception as e:
        sys.exit(f"Dataset read error: {e}")
    
    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    
    X = []
    Y = []
    for i in range(len(tokens)-1):
        if tokens[i] in token_to_id and tokens[i+1] in token_to_id:
            X.append(token_to_id[tokens[i]])
            Y.append(token_to_id[tokens[i+1]])
    
    Y_one_hot = np.zeros((len(Y), vocab_size), dtype=np.float32)
    for i, target_id in enumerate(Y):
        Y_one_hot[i, target_id] = 1.0
    
    return np.array(X), Y_one_hot, token_to_id


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return np.mean((pred - target)**2)


def compute_gradient_for_parameter(args):
    import logging
    import traceback
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.CRITICAL)
    try:
        (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads, X, Y, original_params, i) = args
        
        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            sim_method=sim_method,
            num_threads=num_threads
        )
        model.set_all_parameters(original_params)
        
        shifted_params_plus = original_params.copy()
        shifted_params_plus[i] += np.pi/2
        model.set_all_parameters(shifted_params_plus)
        loss_plus = np.mean([mse_loss(model.forward([x]), y) for x, y in zip(X, Y)])
        
        shifted_params_minus = original_params.copy()
        shifted_params_minus[i] -= np.pi/2
        model.set_all_parameters(shifted_params_minus)
        loss_minus = np.mean([mse_loss(model.forward([x]), y) for x, y in zip(X, Y)])
        
        # Reset to original
        model.set_all_parameters(original_params)
        
        gradient = (loss_plus - loss_minus)/2.0
        return i, gradient
    except Exception as e:
        logging.critical(f"Gradient computation error: {traceback.format_exc()}")
        return i, 0.0


def compute_gradients_parallel(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, num_processes: int = 1) -> np.ndarray:
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
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(compute_gradient_for_parameter, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                i, gradient = future.result()
                gradients[i] = gradient
            except Exception:
                logging.critical(f"Error retrieving gradient: {traceback.format_exc()}")
                gradients[i] = 0.0
    
    return gradients


def train_model_parallel(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray,
                         epochs: int = 10, lr: float = 0.1, num_threads: int = 1,
                         log_callback=None, stop_flag=None, time_lock: threading.Lock = None, time_data=None):
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
        
        params = model.get_all_parameters()
        params -= lr * gradients
        model.set_all_parameters(params)
        
        try:
            total_loss = np.mean([mse_loss(model.forward([x]), y) for x, y in zip(X, Y)])
        except Exception as e:
            logging.error(f"Error computing avg loss: {e}")
            total_loss = float('inf')
        
        if log_callback:
            log_callback(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss:.6f}\n")
        
        # Update time data with thread safety
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


class QELM_GUI:
    def __init__(self, master):
        self.master = master
        master.title("QELM Trainer - Enhanced")
        master.geometry("1400x900")
        master.resizable(False, False)
        
        self.vocab_size = 256
        self.embed_dim = 16
        self.num_heads = 2
        self.hidden_dim = 32
        self.sim_method = 'cpu'
        self.num_threads = min(4, multiprocessing.cpu_count())
        self.model = QuantumLanguageModel(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim,
                                          sim_method=self.sim_method, num_threads=self.num_threads)
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.stop_flag = threading.Event()
        self.time_data = {'start_time': 0, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
        self.time_lock = threading.Lock()
        
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
        self.epochs_entry.insert(0, "2")
        self.epochs_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.05")
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
        
        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        # =======================
        # Run Inference Tab
        # =======================
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)
        
        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        # =======================
        # Manage Token Mappings Tab
        # =======================
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)
        
        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
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
        method = self.sim_method_var.get()
        # Removed thread limitations based on method to allow flexible selection
        max_threads = multiprocessing.cpu_count()
        self.num_threads_spinbox.config(to=max_threads)
        if self.num_threads_var.get() > max_threads:
            self.num_threads_var.set(max_threads)
    
    def log_train(self, message: str):
        self.train_log.config(state='normal')
        self.train_log.insert(tk.END, message)
        self.train_log.see(tk.END)
        self.train_log.config(state='disabled')
    
    def log_infer(self, message: str):
        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')
    
    def log_token_map(self, message: str):
        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')
    
    def select_dataset(self):
        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
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
        
        if hasattr(self, 'dataset_path') and self.dataset_path:
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
            X, Y = create_synthetic_dataset(self.vocab_size, num_samples=200)  # More samples for more noticeable time
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")
        
        self.model.sim_method = sim_method
        self.model.num_threads = num_threads
        self.model.attn.sim_method = sim_method
        self.model.attn.num_threads = num_threads
        self.model.attn.backend = self.model.attn.initialize_simulator()
        
        self.model.ffn.sim_method = sim_method
        self.model.ffn.num_threads = num_threads
        self.model.ffn.backend = self.model.ffn.initialize_simulator()
        
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
        
        training_thread = threading.Thread(target=self.training_process, args=(epochs, lr, num_threads))
        training_thread.start()
    
    def training_process(self, epochs: int, lr: float, num_threads: int):
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
            
            train_model_parallel(
                self.model,
                self.X,
                self.Y,
                epochs=epochs,
                lr=lr,
                num_threads=num_threads,
                log_callback=log_callback,
                stop_flag=self.stop_flag,
                time_lock=self.time_lock,
                time_data=self.time_data
            )
            if not self.stop_flag.is_set():
                self.log_train("Training completed successfully.\n")
                messagebox.showinfo("Training Completed", "Model training completed successfully.")
        except Exception as e:
            err_msg = f"Training error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Training Error", err_msg)
        finally:
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.infer_button.config(state='normal')
            if not self.stop_flag.is_set():
                self.progress['value'] = 100
    
    def stop_training(self):
        self.stop_flag.set()
        self.log_train("Stop signal sent. Will stop after current epoch.\n")
    
    def hard_stop(self):
        self.log_train("Hard stop invoked. Terminating immediately.\n")
        # Immediately terminate the process
        os._exit(0)
    
    def save_model(self):
        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm", filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
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
        try:
            load_path = filedialog.askopenfilename(title="Load Model", filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
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
        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return
        
        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}'...\n")
        
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token,))
        inference_thread.start()
    
    def inference_process(self, input_token: str):
        try:
            run_inference(self.model, input_token, self.token_to_id, self.id_to_token, log_callback=self.log_infer)
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')
    
    def load_token_map(self):
        try:
            file_path = filedialog.askopenfilename(title="Load Token Map", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
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
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')
    
    def update_progress(self, percentage):
        self.progress['value'] = percentage
        self.master.update_idletasks()
    
    def update_resource_usage(self):
        if psutil:
            cpu_usage = f"{psutil.cpu_percent()}%"
        else:
            cpu_usage = "psutil not installed"
        
        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")
        
        self.master.after(1000, self.update_resource_usage)
    
    def update_time_label(self):
        with self.time_lock:
            elapsed = time.time() - self.time_data['start_time'] if self.time_data['start_time'] > 0 else 0
            elapsed_str = f"{elapsed:.1f}s"
            
            if self.time_data['epochs_done'] == 0 and self.time_data['epochs'] > 0:
                remaining_str = "Estimating..."
            else:
                remaining = self.time_data.get('remaining', 0)
                if remaining > 0:
                    remaining_str = f"{remaining:.1f}s"
                else:
                    if self.time_data['epochs_done'] > 0 and self.time_data['epochs_done'] < self.time_data['epochs']:
                        remaining_str = "Estimating..."
                    else:
                        remaining_str = "0s"
        
        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
        
        # Update every second
        self.master.after(1000, self.update_time_label)


def run_inference(model: QuantumLanguageModel, input_token: str, token_to_id: Dict[str, int], id_to_token: Dict[int, str], log_callback=None):
    if not token_to_id:
        raise ValueError("Token mapping is not loaded.")
    
    if input_token not in token_to_id:
        raise ValueError(f"Input token '{input_token}' is not in the token mapping.")
    
    input_id = token_to_id[input_token]
    logits = model.forward([input_id])
    
    # Apply softmax to logits to get probabilities
    probabilities = softmax(logits)
    
    # Get the top 5 tokens
    top_indices = probabilities.argsort()[-5:][::-1]
    top_tokens = [id_to_token.get(idx, "<UNK>") for idx in top_indices]
    top_probs = [probabilities[idx] for idx in top_indices]
    
    response = " | ".join([f"{token} ({prob:.2f})" for token, prob in zip(top_tokens, top_probs)])
    
    if log_callback:
        log_callback(f"Input Token: {input_token} (ID: {input_id})\n")
        log_callback(f"Top Predictions:\n{response}\n\n")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_model(model: QuantumLanguageModel, save_path: str, log_callback=None):
    model.save_model(save_path)
    if log_callback:
        log_callback(f"Model saved to {save_path}\n")


def load_model(model: QuantumLanguageModel, load_path: str, log_callback=None):
    model.load_model(load_path)
    if log_callback:
        log_callback(f"Model loaded from {load_path}\n")


def main():
    freeze_support()
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
