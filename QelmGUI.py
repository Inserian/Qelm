#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================================================
Quantum-Enhanced Language Model (QELM) - Enhanced Complete Training Script with Tkinter GUI
====================================================================================================

This script defines an advanced Quantum-Enhanced Language Model (QELM) with the following features:
1. Gradient-Based Optimization using the Parameter Shift Rule.
2. Advanced Quantum Circuit Design with entangling gates and multiple layers.
3. Support for both Synthetic and Real Datasets resembling language data.
4. Enhanced Model Architecture with residual connections and layer normalization.
5. Robust Parameter Persistence with versioning and validation using a custom .qelm file extension.
6. User-Friendly Graphical User Interface (GUI) using Tkinter for training, inference, saving, loading, and exploring token mappings.

Dependencies:
- qiskit
- qiskit-aer
- numpy
- scipy
- nltk
- tkinter

Ensure all dependencies are installed before running the script.

Remember to update qiskit calls with correct versions with each update. (Not sure why they change it)

====================================================================================================
"""

import sys
import numpy as np
import json
import logging
from typing import List, Dict
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading

# Initialize NLTK data (only the first time)
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================
# Utility Functions
# ============================

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


# ============================
# Quantum Parameter Store
# ============================

class QuantumParameterStore:
    """
    Stores parameters for quantum gates.
    """
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


# ============================
# Quantum Attention Layer
# ============================

class QuantumAttentionLayer:
    """
    Quantum-enhanced attention layer with advanced circuit design.
    """
    def __init__(self, embed_dim: int, num_heads: int, prefix: str = "attn"):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            sys.exit("Error: embed_dim must be divisible by num_heads.")
        
        # Initialize parameter stores
        self.query_params = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_Q")
        self.key_params   = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_K")
        self.value_params = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_V")
        self.out_params   = QuantumParameterStore(embed_dim * embed_dim, prefix=f"{prefix}_O")
        
        # Initialize quantum simulator
        self.backend = AerSimulator(method='statevector', max_parallel_threads=32)
    
    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        """
        Build the quantum circuit with entangling gates and multiple layers.
        """
        qubits_needed = max(1, int(np.ceil(np.log2(self.embed_dim))))
        circuit = QuantumCircuit(qubits_needed)
        
        # Initialize the quantum state
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        if self.embed_dim > 0:
            state_prep_vec[:self.embed_dim] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, qubits=range(qubits_needed))
        
        # Apply parameterized rotations with entangling gates
        num_layers = 2  # Multiple layers
        for layer in range(num_layers):
            # Parameterized RY rotations
            for i in range(qubits_needed):
                theta = param_store.values[layer * qubits_needed + i]
                circuit.ry(theta, i)
            
            # Entangling CNOT gates
            for i in range(qubits_needed - 1):
                circuit.cx(i, i+1)
        
        # Final RY rotations
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)
        
        circuit.save_statevector()
        return circuit
    
    def forward(self, input_vector: np.ndarray, mode: str = 'query') -> np.ndarray:
        """
        Perform a forward pass through the quantum attention layer.
        """
        input_vector = normalize_vector(input_vector)
        if mode == 'query':
            circuit = self.build_circuit(input_vector, self.query_params)
        elif mode == 'key':
            circuit = self.build_circuit(input_vector, self.key_params)
        elif mode == 'value':
            circuit = self.build_circuit(input_vector, self.value_params)
        elif mode == 'out':
            circuit = self.build_circuit(input_vector, self.out_params)
        else:
            sys.exit("Error: Invalid mode for QuantumAttentionLayer.forward")
        
        # Simulate the circuit
        try:
            job = self.backend.run(circuit, shots=1024)
            result = job.result()
            final_state = result.get_statevector(circuit)
        except Exception as e:
            logging.error(f"An error occurred during quantum simulation: {e}")
            sys.exit(1)
        
        # Extract and normalize the output vector
        output_length = self.embed_dim
        if len(final_state.data) < output_length:
            logging.warning(f"Final state vector length ({len(final_state.data)}) is less than embed_dim ({output_length}). Padding with zeros.")
            output_vec = np.real(final_state.data[:len(final_state.data)])  # Use available data
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state.data[:output_length])
        
        return normalize_vector(output_vec)
    
    def get_all_parameters(self) -> np.ndarray:
        """
        Get all parameters as a single array.
        """
        return np.concatenate([
            self.query_params.get_values(),
            self.key_params.get_values(),
            self.value_params.get_values(),
            self.out_params.get_values()
        ])
    
    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single array.
        """
        total_size = self.query_params.size + self.key_params.size + self.value_params.size + self.out_params.size
        if params.shape[0] != total_size:
            sys.exit("Error: Parameter size mismatch in QuantumAttentionLayer.")
        q_size = self.query_params.size
        k_size = self.key_params.size
        v_size = self.value_params.size
        o_size = self.out_params.size
        self.query_params.set_values(params[:q_size])
        self.key_params.set_values(params[q_size:q_size + k_size])
        self.value_params.set_values(params[q_size + k_size:q_size + k_size + v_size])
        self.out_params.set_values(params[q_size + k_size + v_size:])
    
    def to_dict(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "query_params": self.query_params.to_dict(),
            "key_params": self.key_params.to_dict(),
            "value_params": self.value_params.to_dict(),
            "out_params": self.out_params.to_dict()
        }
    
    def from_dict(self, d: dict):
        if d["embed_dim"] != self.embed_dim or d["num_heads"] != self.num_heads:
            sys.exit("Error: Attention layer configuration mismatch.")
        self.query_params.from_dict(d["query_params"])
        self.key_params.from_dict(d["key_params"])
        self.value_params.from_dict(d["value_params"])
        self.out_params.from_dict(d["out_params"])


# ============================
# Quantum Feed-Forward Layer
# ============================

class QuantumFeedForwardLayer:
    """
    Quantum-enhanced feed-forward layer with advanced circuit design.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, prefix: str = "ffn"):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Initialize parameter stores
        self.w1_params = QuantumParameterStore(embed_dim * hidden_dim, prefix=f"{prefix}_W1")
        self.w2_params = QuantumParameterStore(hidden_dim * embed_dim, prefix=f"{prefix}_W2")
        
        # Initialize quantum simulator
        self.backend = AerSimulator(method='statevector', max_parallel_threads=32)
    
    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        """
        Build the quantum circuit with entangling gates and multiple layers.
        """
        qubits_needed = max(1, int(np.ceil(np.log2(self.hidden_dim))))
        circuit = QuantumCircuit(qubits_needed)
        
        # Initialize the quantum state
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        if self.embed_dim > 0:
            state_prep_vec[:self.embed_dim] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, qubits=range(qubits_needed))
        
        # Apply parameterized rotations with entangling gates
        num_layers = 2  # Multiple layers
        for layer in range(num_layers):
            # Parameterized RY rotations
            for i in range(qubits_needed):
                theta = param_store.values[layer * qubits_needed + i]
                circuit.ry(theta, i)
            
            # Entangling CNOT gates
            for i in range(qubits_needed - 1):
                circuit.cx(i, i+1)
        
        # Final RY rotations
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)
        
        circuit.save_statevector()
        return circuit
    
    def forward(self, input_vector: np.ndarray, layer: str = 'w1') -> np.ndarray:
        """
        Perform a forward pass through the quantum feed-forward layer.
        """
        input_vector = normalize_vector(input_vector)
        if layer == 'w1':
            circuit = self.build_circuit(input_vector, self.w1_params)
        elif layer == 'w2':
            circuit = self.build_circuit(input_vector, self.w2_params)
        else:
            sys.exit("Error: Invalid layer for QuantumFeedForwardLayer.forward")
        
        # Simulate the circuit
        try:
            job = self.backend.run(circuit, shots=1024)
            result = job.result()
            final_state = result.get_statevector(circuit)
        except Exception as e:
            logging.error(f"An error occurred during quantum simulation: {e}")
            sys.exit(1)
        
        # Extract and normalize the output vector
        output_length = self.hidden_dim
        if len(final_state.data) < output_length:
            logging.warning(f"Final state vector length ({len(final_state.data)}) is less than hidden_dim ({output_length}). Padding with zeros.")
            output_vec = np.real(final_state.data[:len(final_state.data)])  # Use available data
            output_vec = np.pad(output_vec, (0, output_length - len(output_vec)), 'constant')
        else:
            output_vec = np.real(final_state.data[:output_length])
        
        return normalize_vector(output_vec)
    
    def get_all_parameters(self) -> np.ndarray:
        """
        Get all parameters as a single array.
        """
        return np.concatenate([
            self.w1_params.get_values(),
            self.w2_params.get_values()
        ])
    
    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single array.
        """
        total_size = self.w1_params.size + self.w2_params.size
        if params.shape[0] != total_size:
            sys.exit("Error: Parameter size mismatch in QuantumFeedForwardLayer.")
        w1_size = self.w1_params.size
        self.w1_params.set_values(params[:w1_size])
        self.w2_params.set_values(params[w1_size:])
    
    def to_dict(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "w1_params": self.w1_params.to_dict(),
            "w2_params": self.w2_params.to_dict()
        }
    
    def from_dict(self, d: dict):
        if d["embed_dim"] != self.embed_dim or d["hidden_dim"] != self.hidden_dim:
            sys.exit("Error: Feed-forward layer configuration mismatch.")
        self.w1_params.from_dict(d["w1_params"])
        self.w2_params.from_dict(d["w2_params"])


# ============================
# Quantum Language Model
# ============================

class QuantumLanguageModel:
    """
    Quantum-Enhanced Language Model combining attention and feed-forward layers with a Projection Layer.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Initialize embeddings
        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        
        # Initialize quantum layers
        self.attn = QuantumAttentionLayer(embed_dim, num_heads, prefix="layer1_attn")
        self.ffn  = QuantumFeedForwardLayer(embed_dim, hidden_dim, prefix="layer1_ffn")
        
        # Initialize projection layer
        self.W_proj = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.01  # Projection matrix
        
        # Initialize output weights
        self.W_out = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01  # Output matrix
        
        # Initialize quantum parameters
        self._initialize_quantum_params()
    
    def _initialize_quantum_params(self):
        """
        Randomly initialize quantum parameters.
        """
        scale = 0.1  # Increased scale for better parameter exploration
        self.attn.query_params.set_values(np.random.randn(self.attn.query_params.size) * scale)
        self.attn.key_params.set_values(np.random.randn(self.attn.key_params.size) * scale)
        self.attn.value_params.set_values(np.random.randn(self.attn.value_params.size) * scale)
        self.attn.out_params.set_values(np.random.randn(self.attn.out_params.size) * scale)
        self.ffn.w1_params.set_values(np.random.randn(self.ffn.w1_params.size) * scale)
        self.ffn.w2_params.set_values(np.random.randn(self.ffn.w2_params.size) * scale)
        # W_proj and W_out are already initialized in __init__
    
    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        """
        Perform a forward pass through the entire model.
        """
        if not input_ids:
            sys.exit("Error: input_ids list is empty.")
        
        # Embedding lookup
        try:
            x = self.embeddings[input_ids[0]]
        except IndexError:
            sys.exit(f"Error: input_id {input_ids[0]} is out of bounds for vocabulary size {self.vocab_size}.")
        
        # Quantum attention
        attn_output = self.attn.forward(x, mode='query')
        key_output = self.attn.forward(x, mode='key')
        value_output = self.attn.forward(x, mode='value')
        
        # Combine attention outputs (placeholder for actual attention mechanism)
        combined_attn = attn_output + key_output + value_output
        
        if use_residual:
            if x.shape[0] != combined_attn.shape[0]:
                sys.exit(f"Error: Shape mismatch in residual connection. x shape: {x.shape}, combined_attn shape: {combined_attn.shape}")
            x = normalize_vector(x + combined_attn)  # Residual connection and normalization
        else:
            x = combined_attn
        
        # Quantum feed-forward
        ffn_output_w1 = self.ffn.forward(x, layer='w1')  # Shape: (hidden_dim,)
        ffn_output_w2 = self.ffn.forward(x, layer='w2')  # Shape: (hidden_dim,)
        ffn_output = ffn_output_w1 + ffn_output_w2      # Shape: (hidden_dim,)
        
        if use_residual:
            # Project ffn_output to embed_dim
            ffn_output_proj = self.W_proj @ ffn_output  # Shape: (embed_dim,)
            # Ensure x and ffn_output_proj have the same shape
            if x.shape[0] != ffn_output_proj.shape[0]:
                sys.exit(f"Error: Shape mismatch in residual connection after projection. x shape: {x.shape}, ffn_output_proj shape: {ffn_output_proj.shape}")
            x = normalize_vector(x + ffn_output_proj)  # Residual connection and normalization
        else:
            x = ffn_output
        
        # Output logits (linear transformation)
        logits = self.W_out @ x  # Shape: (vocab_size,)
        return logits
    
    def get_all_parameters(self) -> np.ndarray:
        """
        Get all quantum parameters concatenated into a single array, including W_proj and W_out.
        """
        return np.concatenate([
            self.attn.get_all_parameters(),
            self.ffn.get_all_parameters(),
            self.W_proj.flatten(),
            self.W_out.flatten()
        ])
    
    def set_all_parameters(self, params: np.ndarray):
        """
        Set all quantum parameters from a single array, including W_proj and W_out.
        """
        attn_size = self.attn.query_params.size + self.attn.key_params.size + self.attn.value_params.size + self.attn.out_params.size
        ffn_size = self.ffn.w1_params.size + self.ffn.w2_params.size
        proj_size = self.embed_dim * self.hidden_dim
        out_size = self.vocab_size * self.embed_dim
        expected_size = attn_size + ffn_size + proj_size + out_size
        if params.shape[0] != expected_size:
            sys.exit(f"Error: Parameter size mismatch in QuantumLanguageModel. Expected {expected_size}, got {params.shape[0]}.")
        
        # Set attention parameters
        attn_params = params[:attn_size]
        self.attn.set_all_parameters(attn_params)
        
        # Set feed-forward parameters
        ffn_params = params[attn_size:attn_size + ffn_size]
        self.ffn.set_all_parameters(ffn_params)
        
        # Set projection matrix
        proj_params = params[attn_size + ffn_size:attn_size + ffn_size + proj_size]
        self.W_proj = proj_params.reshape(self.embed_dim, self.hidden_dim)
        
        # Set output weights
        out_params = params[attn_size + ffn_size + proj_size:]
        self.W_out = out_params.reshape(self.vocab_size, self.embed_dim)
    
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
            sys.exit("Error: Model configuration in file does not match this QLM instance.")
        
        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)
    
    def save_model(self, save_path: str):
        """
        Save model parameters (embeddings and quantum parameters) to a .qelm file.
        """
        model_dict = self.to_dict()
        try:
            with open(save_path, 'w') as f:
                json.dump(model_dict, f)
            logging.info(f"Model saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            sys.exit(1)
    
    def load_model(self, load_path: str):
        """
        Load model parameters (embeddings and quantum parameters) from a .qelm file.
        """
        try:
            with open(load_path, 'r') as f:
                model_dict = json.load(f)
        except FileNotFoundError:
            sys.exit(f"Error: The file {load_path} does not exist.")
        except json.JSONDecodeError:
            sys.exit(f"Error: The file {load_path} is not a valid JSON file.")
        except Exception as e:
            sys.exit(f"Error reading the model file: {e}")
        
        # Version check
        if "version" not in model_dict or model_dict["version"] != "1.0":
            sys.exit("Error: Unsupported model version.")
        
        try:
            self.from_dict(model_dict)
            logging.info(f"Model loaded from {load_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            sys.exit(1)


# ============================
# Synthetic Dataset
# ============================

def create_synthetic_dataset(vocab_size: int, num_samples: int = 100):
    """
    Create a synthetic dataset for demonstration:
    Each sample: input_token -> random token, target -> one-hot vector
    """
    X = np.random.randint(0, vocab_size, size=(num_samples,))
    Y = np.zeros((num_samples, vocab_size), dtype=np.float32)
    for i in range(num_samples):
        # Create a "target" as a random one-hot vector different from input token
        target_id = np.random.randint(0, vocab_size)
        Y[i, target_id] = 1.0
    return X, Y


# ============================
# Real Dataset Loader (Optional)
# ============================

def load_real_dataset(file_path: str, vocab_size: int):
    """
    Load and preprocess a real language dataset.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        sys.exit(f"Error reading the dataset file: {e}")
    
    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    
    # Select top vocab_size tokens
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
    token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    
    # Convert tokens to IDs
    X = []
    Y = []
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        if current_token in token_to_id and next_token in token_to_id:
            X.append(token_to_id[current_token])
            Y.append(token_to_id[next_token])
    
    # One-hot encode targets
    Y_one_hot = np.zeros((len(Y), vocab_size), dtype=np.float32)
    for i, target_id in enumerate(Y):
        Y_one_hot[i, target_id] = 1.0
    
    return np.array(X), Y_one_hot, token_to_id


# ============================
# Loss Function
# ============================

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss between prediction and target.
    """
    return np.mean((pred - target)**2)


# ============================
# Training Functions
# ============================

def compute_gradients(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute gradients of the loss with respect to all quantum parameters using the Parameter Shift Rule.
    Note: This is a simplified implementation for demonstration purposes.
    """
    gradients = np.zeros_like(model.get_all_parameters())
    original_params = model.get_all_parameters().copy()
    
    for i in range(len(original_params)):
        # Shift parameter positively
        shifted_params_plus = original_params.copy()
        shifted_params_plus[i] += np.pi / 2
        model.set_all_parameters(shifted_params_plus)
        loss_plus = 0.0
        for x, y in zip(X, Y):
            logits = model.forward([x])
            loss_plus += mse_loss(logits, y)
        loss_plus /= len(X)
        
        # Shift parameter negatively
        shifted_params_minus = original_params.copy()
        shifted_params_minus[i] -= np.pi / 2
        model.set_all_parameters(shifted_params_minus)
        loss_minus = 0.0
        for x, y in zip(X, Y):
            logits = model.forward([x])
            loss_minus += mse_loss(logits, y)
        loss_minus /= len(X)
        
        # Reset to original parameter
        model.set_all_parameters(original_params)
        
        # Compute gradient using Parameter Shift Rule
        gradients[i] = (loss_plus - loss_minus) / 2
    
    return gradients

def train_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, epochs: int = 10, lr: float = 0.1, log_callback=None):
    """
    Train the model using gradient-based optimization with the Parameter Shift Rule.
    """
    for epoch in range(epochs):
        logging.info(f"Starting Epoch {epoch+1}/{epochs}")
        if log_callback:
            log_callback(f"Starting Epoch {epoch+1}/{epochs}\n")
        
        # Compute gradients
        gradients = compute_gradients(model, X, Y)
        
        # Update parameters
        params = model.get_all_parameters()
        params -= lr * gradients
        model.set_all_parameters(params)
        
        # Compute average loss
        total_loss = 0.0
        for x, y in zip(X, Y):
            logits = model.forward([x])
            loss = mse_loss(logits, y)
            total_loss += loss
        avg_loss = total_loss / len(X)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        if log_callback:
            log_callback(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}\n")
    
    logging.info("Training completed.")
    if log_callback:
        log_callback("Training completed.\n")


# ============================
# Parameter Persistence
# ============================

def save_model(model: QuantumLanguageModel, save_path: str, log_callback=None):
    """
    Save model parameters (embeddings and quantum parameters) to a .qelm file.
    """
    model.save_model(save_path)
    if log_callback:
        log_callback(f"Model saved to {save_path}\n")


def load_model(model: QuantumLanguageModel, load_path: str, log_callback=None):
    """
    Load model parameters (embeddings and quantum parameters) from a .qelm file.
    """
    model.load_model(load_path)
    if log_callback:
        log_callback(f"Model loaded from {load_path}\n")


# ============================
# Inference Function
# ============================

def run_inference(model: QuantumLanguageModel, input_token: str, token_to_id: Dict[str, int], id_to_token: Dict[int, str], log_callback=None):
    """
    Run a forward pass of the model and print the logits.
    """
    if input_token not in token_to_id:
        output = f"Token '{input_token}' not found in the vocabulary.\n"
        print(output)
        if log_callback:
            log_callback(output)
        return
    
    input_id = token_to_id[input_token]
    logits = model.forward([input_id])
    predicted_id = np.argmax(logits)
    response = id_to_token.get(predicted_id, "unknown")
    output = f"Input Token: '{input_token}' (ID: {input_id})\n"
    output += f"Predicted Token ID: {predicted_id}\n"
    output += f"Predicted Token: '{response}'\n"
    output += f"Logits: {logits}\n"
    print(output)
    if log_callback:
        log_callback(output)


# ============================
# GUI Implementation with Tkinter
# ============================

class QELM_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Quantum-Enhanced Language Model (QELM) Trainer")
        master.geometry("1000x800")
        master.resizable(False, False)
        
        # Initialize model
        self.vocab_size = 256
        self.embed_dim = 16
        self.num_heads = 2
        self.hidden_dim = 32
        self.model = QuantumLanguageModel(self.vocab_size, self.embed_dim, self.num_heads, self.hidden_dim)
        
        # Initialize token mappings
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Create GUI components
        self.create_widgets()
    
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Tabs
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        
        # ============================
        # Train Model Tab
        # ============================
        
        # Dataset Selection
        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)
        
        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset).pack(side='right', padx=10, pady=10)
        
        # Hyperparameters
        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Hyperparameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)
        
        # Epochs
        ttk.Label(hyperparams_frame, text="Epochs:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.epochs_entry = ttk.Entry(hyperparams_frame, width=15)
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        # Learning Rate
        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_frame, width=15)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        
        # Training Controls
        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)
        
        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)
        
        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.tab_train, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Training Log
        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10))
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ============================
        # Run Inference Tab
        # ============================
        
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)
        
        # Input Token
        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30)
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        
        # Inference Controls
        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)
        
        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)
        
        # Inference Log
        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10))
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ============================
        # Manage Token Mappings Tab
        # ============================
        
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Load Token Map
        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)
        
        # Display Token Mappings
        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10))
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)
    
    def log_train(self, message: str):
        """
        Append message to the training log.
        """
        self.train_log.config(state='normal')
        self.train_log.insert(tk.END, message)
        self.train_log.see(tk.END)
        self.train_log.config(state='disabled')
    
    def log_infer(self, message: str):
        """
        Append message to the inference log.
        """
        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')
    
    def log_token_map(self, message: str):
        """
        Append message to the token map display.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')
    
    def select_dataset(self):
        """
        Open a file dialog to select a dataset file.
        """
        file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_path:
            self.dataset_path = file_path
            self.dataset_path_var.set(file_path)
            self.log_train(f"Selected Dataset: {file_path}\n")
            # Reset token mappings if a new dataset is selected
            self.token_to_id = {}
            self.id_to_token = {}
    
    def train_model(self):
        """
        Start the training process in a separate thread.
        """
        # Retrieve hyperparameters
        try:
            epochs = int(self.epochs_entry.get())
            lr = float(self.lr_entry.get())
            if epochs <= 0 or lr <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive numbers for epochs and learning rate.")
            return
        
        # Check if dataset is selected
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
                messagebox.showerror("Error", f"Failed to load dataset: {e}")
                return
        else:
            # Use synthetic dataset
            X, Y = create_synthetic_dataset(self.vocab_size, num_samples=100)
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")
        
        # Disable the train button to prevent multiple clicks
        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.progress['value'] = 0
        self.log_train("Starting training...\n")
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=self.training_process, args=(epochs, lr))
        training_thread.start()
    
    def training_process(self, epochs: int, lr: float):
        """
        The actual training process.
        """
        try:
            train_model(self.model, self.X, self.Y, epochs=epochs, lr=lr, log_callback=self.log_train)
            self.log_train("Training process completed.\n")
            messagebox.showinfo("Training Completed", "Model training has been completed successfully.")
        except Exception as e:
            self.log_train(f"An error occurred during training: {e}\n")
            messagebox.showerror("Training Error", f"An error occurred during training: {e}")
        finally:
            # Re-enable the buttons
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.progress['value'] = 100
    
    def save_model(self):
        """
        Save the model to a .qelm file.
        """
        save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm", filetypes=(("QELM Files", "*.qelm"), ("All Files", "*.*")))
        if save_path:
            try:
                save_model(self.model, save_path, log_callback=self.log_train)
                # Save token mappings if available
                if self.token_to_id:
                    token_map_path = save_path.replace(".qelm", "_token_map.json")
                    with open(token_map_path, 'w') as f:
                        json.dump(self.token_to_id, f, indent=4)
                    self.log_train(f"Token mappings saved to {token_map_path}\n")
                messagebox.showinfo("Model Saved", f"Model saved successfully to {save_path}")
            except Exception as e:
                self.log_train(f"Failed to save model: {e}\n")
                messagebox.showerror("Save Error", f"Failed to save model: {e}")
    
    def load_model(self):
        """
        Load the model from a .qelm file.
        """
        load_path = filedialog.askopenfilename(title="Load Model", filetypes=(("QELM Files", "*.qelm"), ("All Files", "*.*")))
        if load_path:
            try:
                load_model(self.model, load_path, log_callback=self.log_train)
                # Load token mappings if available
                token_map_path = load_path.replace(".qelm", "_token_map.json")
                try:
                    with open(token_map_path, 'r') as f:
                        self.token_to_id = json.load(f)
                    self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                    self.log_train(f"Loaded token mappings from {token_map_path}\n")
                    self.display_token_map()
                except FileNotFoundError:
                    self.log_train("Token mappings file not found. Inference may be limited.\n")
                messagebox.showinfo("Model Loaded", f"Model loaded successfully from {load_path}")
            except Exception as e:
                self.log_train(f"Failed to load model: {e}\n")
                messagebox.showerror("Load Error", f"Failed to load model: {e}")
    
    def run_inference(self):
        """
        Run inference based on user input.
        """
        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return
        
        # Disable the inference button to prevent multiple clicks
        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for Input Token: '{input_token}'\n")
        
        # Start inference in a separate thread
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token,))
        inference_thread.start()
    
    def inference_process(self, input_token: str):
        """
        The actual inference process.
        """
        try:
            run_inference(self.model, input_token, self.token_to_id, self.id_to_token, log_callback=self.log_infer)
            messagebox.showinfo("Inference Completed", "Model inference has been completed successfully.")
        except Exception as e:
            self.log_infer(f"An error occurred during inference: {e}\n")
            messagebox.showerror("Inference Error", f"An error occurred during inference: {e}")
        finally:
            # Re-enable the inference button
            self.infer_button.config(state='normal')
    
    def load_token_map(self):
        """
        Load and display token mappings from a JSON file.
        """
        file_path = filedialog.askopenfilename(title="Load Token Map", filetypes=(("JSON Files", "*.json"), ("All Files", "*.*")))
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded successfully from {file_path}")
            except Exception as e:
                self.log_token_map(f"Failed to load token map: {e}\n")
                messagebox.showerror("Load Error", f"Failed to load token map: {e}")
    
    def display_token_map(self):
        """
        Display all token mappings in the token map display.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')


# ============================
# Main Execution
# ============================

def main():
    root = tk.Tk()
    gui = QELM_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()