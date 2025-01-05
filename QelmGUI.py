#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
import logging
import traceback
import threading
import multiprocessing
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Callable
import queue
import subprocess
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

try:
    import psutil
except ImportError:
    psutil = None

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

nltk.download('punkt', quiet=True)

# Configure Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec.copy()
    return vec / norm


# --------------------------------------------------------------------
# Spin Measurement Integration (Four Decimal Precision)
# --------------------------------------------------------------------
def measure_qubit_spin_z(qc: "QuantumChannel") -> float:
    """
    Measure the spin along the Z axis (expectation value) for the quantum circuit in qc.
    Round to 4 decimal places. Returns a float in [-1.0, +1.0].
    """
    temp_circuit = qc.circuit.copy()
    backend = qc.backend
    job = backend.run(temp_circuit)
    result = job.result()
    statevector = result.get_statevector(temp_circuit)
    alpha = np.abs(statevector[0])**2
    beta = np.abs(statevector[1])**2
    z_expectation = alpha - beta
    return round(z_expectation, 4)


# --------------------------------------------------------------------
# Existing Pseudo ExponentialSubwordTokenizer
# --------------------------------------------------------------------
class ExponentialSubwordTokenizer:
    """
    A pseudo subword tokenizer that can exponentially expand the token set
    by identifying common subword units using a rudimentary approach.
    In practice, you might replace this with an advanced technique such
    as Byte-Pair Encoding (BPE), SentencePiece, or WordPiece.

    ENHANCED: 
    - Optional punctuation handling to unify repeated punctuation and separate punctuation from words.
    """
    def __init__(self, vocab_size: int = 1000, min_subword_freq: int = 2, handle_punctuation: bool = True):
        self.vocab_size = vocab_size
        self.min_subword_freq = min_subword_freq
        self.handle_punctuation = handle_punctuation
        self.subword_vocab: Dict[str, int] = {}
        self.special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.is_trained = False

    def train(self, corpus: List[str]):
        from collections import defaultdict
        import re
        if self.handle_punctuation:
            processed_corpus = []
            for sentence in corpus:
                sentence = re.sub(r'([^\w\s])\1+', r'\1', sentence)
                sentence = re.sub(r'([^\w\s])', r' \1 ', sentence)
                processed_corpus.append(sentence)
            corpus = processed_corpus

        char_counts = defaultdict(int)
        for token in self.special_tokens:
            self.subword_vocab[token] = len(self.subword_vocab)

        for sentence in corpus:
            for ch in sentence:
                char_counts[ch] += 1

        sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        for ch, freq in sorted_chars:
            if ch not in self.subword_vocab and len(self.subword_vocab) < self.vocab_size:
                self.subword_vocab[ch] = len(self.subword_vocab)

        merges = True
        while merges and len(self.subword_vocab) < self.vocab_size:
            merges = False
            pair_counts = defaultdict(int)
            for sentence in corpus:
                tokens = self.tokenize_line(sentence, fallback_char_level=True)
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1
            if pair_counts:
                frequent_pair, count = max(pair_counts.items(), key=lambda x: x[1])
                if count >= self.min_subword_freq:
                    new_subword = frequent_pair[0] + frequent_pair[1]
                    if new_subword not in self.subword_vocab:
                        self.subword_vocab[new_subword] = len(self.subword_vocab)
                        merges = True
        self.is_trained = True
        logging.info(f"ExponentialSubwordTokenizer: Trained with vocab size {len(self.subword_vocab)}.")

    def tokenize_line(self, text: str, fallback_char_level: bool = False) -> List[str]:
        if not self.is_trained:
            return list(text)
        tokens = []
        i = 0
        while i < len(text):
            match_found = False
            for length in range(min(self.vocab_size, len(text) - i), 0, -1):
                candidate = text[i : i + length]
                if candidate in self.subword_vocab:
                    tokens.append(candidate)
                    i += length
                    match_found = True
                    break
            if not match_found:
                if fallback_char_level:
                    tokens.append(text[i])
                    i += 1
                else:
                    tokens.append("<UNK>")
                    i += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_line(text, fallback_char_level=True)
        token_ids = [self.subword_vocab.get(tok, self.subword_vocab.get("<UNK>")) for tok in tokens]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.subword_vocab.items()}
        subwords = []
        for tid in token_ids:
            subwords.append(inv_vocab.get(tid, "<UNK>"))
        return "".join(subwords)

    def get_vocab(self) -> Dict[str, int]:
        return self.subword_vocab.copy()

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.subword_vocab.items()}


def load_dataset_with_exponential_tokenizer(file_path: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    tokenizer = ExponentialSubwordTokenizer(vocab_size=vocab_size, handle_punctuation=True)
    tokenizer.train(lines)

    all_ids = []
    next_ids = []
    for line in lines:
        tokens_line = tokenizer.encode(line)
        if len(tokens_line) < 2:
            continue
        for i in range(len(tokens_line) - 1):
            all_ids.append(tokens_line[i])
            next_ids.append(tokens_line[i + 1])

    X = np.array(all_ids, dtype=np.int32)
    Y = np.array(next_ids, dtype=np.int32)
    id_to_token = tokenizer.get_id_to_token_map()
    token_to_id = tokenizer.get_vocab()
    return X, Y, token_to_id, id_to_token


# --------------------------------------------------------------------
# Byte-Pair Encoding (BPE) for improved tokenization
# --------------------------------------------------------------------
class BytePairEncodingTokenizer:
    """
    A more formal BPE tokenizer implementation.
    Iteratively merges frequent pairs of tokens until vocabulary limit
    or until no more merges are beneficial.
    """
    def __init__(self, vocab_size: int = 8000, min_freq_threshold: int = 2):
        self.vocab_size = vocab_size
        self.min_freq_threshold = min_freq_threshold
        self.vocab: Dict[str, int] = {}
        self.merges = []
        self.special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.is_trained = False

    def train(self, corpus: List[str]):
        from collections import defaultdict
        # Initialize vocabulary with special tokens
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Convert each line into a list of characters + special end symbol
        # Example: "cat" -> ['c', 'a', 't', '</w>']
        corpus_tokens = []
        for line in corpus:
            chars = list(line.strip())
            if chars:
                corpus_tokens.append(chars + ["</w>"])  # using </w> to mark word boundaries

        # Count initial frequency of token pairs
        pair_freq = defaultdict(int)
        for tokens in corpus_tokens:
            for i in range(len(tokens) - 1):
                pair_freq[(tokens[i], tokens[i + 1])] += 1

        # Build merges until vocab_size
        while len(self.vocab) < self.vocab_size and pair_freq:
            # find the most frequent pair
            best_pair, freq = max(pair_freq.items(), key=lambda x: x[1])
            if freq < self.min_freq_threshold:
                break
            merged_symbol = best_pair[0] + best_pair[1]

            # add to merges
            if merged_symbol not in self.vocab:
                self.vocab[merged_symbol] = len(self.vocab)
            self.merges.append(best_pair)

            # merge in corpus
            new_pair_freq = defaultdict(int)
            for idx, tokens in enumerate(corpus_tokens):
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        merged = merged_symbol
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                corpus_tokens[idx] = new_tokens
            # recalc frequencies
            for tokens in corpus_tokens:
                for i in range(len(tokens) - 1):
                    new_pair_freq[(tokens[i], tokens[i + 1])] += 1
            pair_freq = new_pair_freq
        self.is_trained = True
        logging.info(f"BPE: Trained with vocab size {len(self.vocab)} merges={len(self.merges)}.")

    def encode(self, line: str) -> List[int]:
        if not self.is_trained:
            # fallback to naive char
            return [self.vocab.get(ch, self.vocab.get("<UNK>")) for ch in list(line)]
        tokens = list(line.strip()) + ["</w>"]
        # merge iteratively
        merges_applied = True
        while merges_applied:
            merges_applied = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    merged_symbol = tokens[i] + tokens[i + 1]
                    new_tokens.append(merged_symbol)
                    i += 2
                    merges_applied = True
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(tok, self.vocab.get("<UNK>")) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        symbols = [inv_vocab.get(tid, "<UNK>") for tid in token_ids]
        word = "".join(symbols).replace("</w>", "")
        return word

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.vocab.items()}


# --------------------------------------------------------------------
# WordPiece Tokenizer (Basic outline)
# --------------------------------------------------------------------
class WordPieceTokenizer:
    """
    WordPiece as used in BERT. Uses a likelihood-based or frequency-based
    approach to split words into subwords. This version is simplified and
    uses a frequency-based method for merges.
    """
    def __init__(self, vocab_size=8000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.vocab: Dict[str, int] = {}
        self.is_trained = False

    def train(self, corpus: List[str]):
        from collections import defaultdict
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        word_freq = defaultdict(int)
        for line in corpus:
            for word in line.split():
                word_freq[word] += 1

        # sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words:
            if freq < self.min_freq:
                continue
            if len(self.vocab) >= self.vocab_size:
                break
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.is_trained = True
        logging.info(f"WordPiece: Trained with vocab size {len(self.vocab)}.")

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # fallback subword splitting
                sub_tokens = self._split_into_subwords(word)
                tokens.extend(self.vocab.get(st, self.vocab.get("<UNK>")) for st in sub_tokens)
        return tokens

    def _split_into_subwords(self, word: str) -> List[str]:
        # basic approach: keep dropping chars
        subwords = []
        start = 0
        while start < len(word):
            for end in range(len(word), start, -1):
                piece = word[start:end]
                if piece in self.vocab:
                    subwords.append(piece)
                    start = end
                    break
            else:
                subwords.append("<UNK>")
                break
        return subwords

    def decode(self, token_ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join(inv_vocab.get(tid, "<UNK>") for tid in token_ids)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {v: k for k, v in self.vocab.items()}


# --------------------------------------------------------------------
# SentencePiece (Skeleton)
# --------------------------------------------------------------------
class SentencePieceTokenizer:
    """
    This is a skeleton example of how one might integrate SentencePiece
    in Python code. Typically, you'd rely on an external library (sentencepiece)
    SentencePiece can handle spaces as characters, and you can choose
    BPE/unigram/etc. behind the scenes.
    """
    def __init__(self, vocab_size=8000, model_type='bpe'):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.is_trained = False
        self.sp = None  # stand-in for an actual sentencepiece processor

    def train(self, corpus: List[str]):
        # with real sentencepiece, you'd write data to a file, then train:
        # spm.SentencePieceTrainer.Train('--input=corpus.txt --model_prefix=m --vocab_size=8000 --model_type=bpe')
        # self.sp = spm.SentencePieceProcessor()
        # self.sp.Load("m.model")
        # This is a placeholder
        self.is_trained = True
        logging.info("SentencePiece: (placeholder) trained.")

    def encode(self, text: str) -> List[int]:
        # placeholder
        if not self.is_trained:
            return [0]*len(text.split())  # naive
        # with real SP: return self.sp.EncodeAsIds(text)
        return [1,2,3]

    def decode(self, token_ids: List[int]) -> str:
        # placeholder
        return "decoded_text"

    def get_vocab(self) -> Dict[str, int]:
        return {"placeholder":0}

    def get_id_to_token_map(self) -> Dict[int, str]:
        return {0: "placeholder"}


# --------------------------------------------------------------------
# Contextual Tokenization (Outline)
# --------------------------------------------------------------------
class ContextualTokenizer:
    """
    Use a preliminary model or embedding to decide token splits.
    Potentially can incorporate quantum state context, etc.
    """
    def __init__(self, base_tokenizer):
        # base_tokenizer could be any of the above (BPE, WordPiece, etc.)
        self.base_tokenizer = base_tokenizer
        self.is_trained = False

    def train(self, corpus: List[str], context_model=None):
        """
        context_model could be a classical or quantum-based model that
        provides context embeddings or hints about word boundaries.
        """
        self.base_tokenizer.train(corpus)
        # potential approach: refine merges using context
        self.is_trained = True
        logging.info("ContextualTokenizer: (placeholder) trained with base tokenizer context.")

    def encode(self, text: str, context_embedding=None) -> List[int]:
        # if context_embedding is used, adapt subword splits
        # for now, just fallback
        return self.base_tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.base_tokenizer.decode(token_ids)


# --------------------------------------------------------------------
# Tokenization with Subword Regularization (high-level placeholder)
# --------------------------------------------------------------------
class SubwordRegularizationTokenizer:
    """
    Adds randomness to tokenization by sometimes picking alternate merges,
    to reduce overfitting. This is typically done with libraries like
    SentencePiece or others that can produce multiple segmentations.

    We will produce multiple possible encodings and choose randomly.
    """
    def __init__(self, base_tokenizer, alpha=0.1):
        self.base_tokenizer = base_tokenizer
        self.alpha = alpha
        self.is_trained = False

    def train(self, corpus: List[str]):
        self.base_tokenizer.train(corpus)
        self.is_trained = True

    def encode(self, text: str) -> List[int]:
        # get normal encoding
        base_ids = self.base_tokenizer.encode(text)
        # artificially produce a "random" variant
        # in a real scenario, you'd have multiple merges from the tokenizer
        # we do a simple random approach
        if np.random.rand() < self.alpha:
            # e.g., replace a random token with <UNK>
            if base_ids:
                idx = np.random.randint(0, len(base_ids))
                base_ids[idx] = self.base_tokenizer.base_tokenizer.vocab.get("<UNK>", 3)
        return base_ids

    def decode(self, token_ids: List[int]) -> str:
        return self.base_tokenizer.decode(token_ids)


# --------------------------------------------------------------------
# Dynamic Vocabulary (simple illustration)
# --------------------------------------------------------------------
class DynamicVocabTokenizer:
    """
    Allows updating the tokenizer with new tokens on the fly, within a limit.
    """
    def __init__(self, base_tokenizer, vocab_update_limit=50):
        self.base_tokenizer = base_tokenizer
        self.vocab_update_limit = vocab_update_limit
        self.updates_made = 0

    def train(self, corpus: List[str]):
        self.base_tokenizer.train(corpus)

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.split():
            if word not in self.base_tokenizer.get_vocab():
                if self.updates_made < self.vocab_update_limit:
                    # add new token
                    current_vocab = self.base_tokenizer.get_vocab()
                    new_id = len(current_vocab)
                    current_vocab[word] = new_id
                    self.base_tokenizer.vocab = current_vocab
                    self.updates_made += 1
            tokens.append(self.base_tokenizer.get_vocab().get(word, self.base_tokenizer.get_vocab().get("<UNK>")))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return self.base_tokenizer.decode(token_ids)


# --------------------------------------------------------------------
# QuantumChannel & Manager
# --------------------------------------------------------------------
class QuantumChannel:
    def __init__(self, label: str = "Qc", decimal_precision: Optional[int] = None):
        self.label = label
        self.value = 0.0
        self.num_qubits = 1
        self.circuit = QuantumCircuit(self.num_qubits)
        self.parameters = [Parameter(f"{self.label}_theta")]
        self.backend = AerSimulator(method='statevector')
        self.decimal_precision = decimal_precision
        logging.info(f"Initialized {self.label} with {self.num_qubits} qubit(s).")

    def encode(self, value: float):
        if self.decimal_precision is not None and self.decimal_precision >= 0:
            self.value = round(float(np.clip(value, 0.0, 1.0)), self.decimal_precision)
        else:
            self.value = float(np.clip(value, 0.0, 1.0))
        theta = float(2 * np.arcsin(np.sqrt(self.value)))
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.ry(theta, 0)
        self.circuit.save_statevector()
        logging.debug(f"{self.label} encoded with value {self.value} (theta={theta:.4f}).")

    def decode(self) -> float:
        job = self.backend.run(self.circuit)
        result = job.result()
        statevector = result.get_statevector(self.circuit)
        alpha = np.abs(statevector[0])**2
        decoded_value = alpha
        logging.debug(f"{self.label} decoded with value {decoded_value:.4f}.")
        return decoded_value

    def apply_gate(self, gate: str, params: Optional[list] = None):
        if gate.upper() == 'RY' and params:
            theta = float(params[0])
            self.circuit.ry(theta, 0)
            logging.debug(f"{self.label}: Applied RY gate with theta={theta:.4f}.")
        elif gate.upper() == 'RZ' and params:
            theta = float(params[0])
            self.circuit.rz(theta, 0)
            logging.debug(f"{self.label}: Applied RZ gate with theta={theta:.4f}.")
        elif gate.upper() == 'CX' and params:
            pass

    def reset(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.reset(0)
        self.circuit.save_statevector()
        self.value = 0.0
        logging.debug(f"{self.label} has been reset.")


class QuantumChannelManager:
    def __init__(self):
        self.channels: List[QuantumChannel] = []
        self.available_indices: List[int] = []
        self.lock = threading.Lock()
        logging.info("Initialized QuantumChannelManager.")

    def create_channels(self, num_channels: int, decimal_precision: Optional[int] = None):
        with self.lock:
            for i in range(num_channels):
                qc = QuantumChannel(label=f"Qc_{len(self.channels)+1}", decimal_precision=decimal_precision)
                self.channels.append(qc)
                self.available_indices.append(len(self.channels) - 1)
                logging.info(f"Created {qc.label}.")

    def allocate_channels(self, num_required: int) -> List[QuantumChannel]:
        with self.lock:
            if num_required > len(self.available_indices):
                raise ValueError("Not enough available Quantum Channels to allocate.")
            allocated = []
            for _ in range(num_required):
                index = self.available_indices.pop(0)
                allocated.append(self.channels[index])
                logging.info(f"Allocated {self.channels[index].label}.")
            return allocated

    def release_channels(self, allocated_channels: List[QuantumChannel]):
        with self.lock:
            for qc in allocated_channels:
                index = self.channels.index(qc)
                if index not in self.available_indices:
                    self.available_indices.append(index)
                    logging.info(f"Released {qc.label} back to pool.")

    def get_all_channels(self) -> List[QuantumChannel]:
        with self.lock:
            return self.channels.copy()


class SubBitDecoder:
    def __init__(self, manager: QuantumChannelManager):
        self.manager = manager
        self.num_qubits = 1
        self.backend = AerSimulator(method='statevector')
        logging.info("Initialized SubBitDecoder.")

    def decode_and_transform(self, 
                             allocated_channels: List[QuantumChannel],
                             transform_function: Callable[[QuantumCircuit, Optional[dict]], QuantumCircuit],
                             params: Optional[dict] = None) -> List[float]:
        aggregated_value = sum(qc.decode() for qc in allocated_channels)
        normalized_value = np.clip(aggregated_value, 0.0, 1.0)
        logging.debug(f"SubBitDecoder: Aggregated value: {aggregated_value} -> {normalized_value}")

        circuit = QuantumCircuit(self.num_qubits)
        theta = float(2 * np.arcsin(np.sqrt(normalized_value)))
        circuit.ry(theta, 0)
        logging.debug(f"SubBitDecoder: Initialized qubit with RY({theta:.4f}).")

        circuit = transform_function(circuit, params)
        circuit.save_statevector()
        job = self.backend.run(circuit)
        result = job.result()
        statevector = result.get_statevector(circuit)
        alpha = np.abs(statevector[0])**2
        beta = np.abs(statevector[1])**2
        logging.debug(f"SubBitDecoder: alpha={alpha:.4f}, beta={beta:.4f}")

        decoded_value = alpha
        logging.debug(f"SubBitDecoder: Decoded value: {decoded_value:.4f}")
        return [decoded_value]


class GroverOracle:
    def __init__(self, target_state: str):
        self.target_state = target_state
        self.num_qubits = len(target_state)
        self.circuit = QuantumCircuit(self.num_qubits)
        self.build_oracle()

    def build_oracle(self):
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                self.circuit.x(i)
        self.circuit.h(self.num_qubits - 1)
        self.circuit.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        self.circuit.h(self.num_qubits - 1)
        for i, bit in enumerate(self.target_state):
            if bit == '0':
                self.circuit.x(i)
        self.circuit.barrier()

    def get_oracle(self) -> QuantumCircuit:
        return self.circuit


class GroverDiffuser:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(self.num_qubits)
        self.build_diffuser()

    def build_diffuser(self):
        self.circuit.h(range(self.num_qubits))
        self.circuit.x(range(self.num_qubits))
        self.circuit.h(self.num_qubits - 1)
        self.circuit.mct(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        self.circuit.h(self.num_qubits - 1)
        self.circuit.x(range(self.num_qubits))
        self.circuit.h(range(self.num_qubits))
        self.circuit.barrier()

    def get_diffuser(self) -> QuantumCircuit:
        return self.circuit


class GroverSearch:
    def __init__(self, target_state: str):
        self.oracle = GroverOracle(target_state)
        self.num_qubits = self.oracle.num_qubits
        self.diffuser = GroverDiffuser(self.num_qubits)
        self.circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.build_circuit()
        self.backend = AerSimulator(method='statevector')

    def build_circuit(self):
        self.circuit.h(range(self.num_qubits))
        iterations = int((np.pi/4) * np.sqrt(2**self.num_qubits))
        for _ in range(iterations):
            self.circuit.append(self.oracle.get_oracle(), range(self.num_qubits))
            self.circuit.append(self.diffuser.get_diffuser(), range(self.num_qubits))
        self.circuit.measure(range(self.num_qubits), range(self.num_qubits))

    def run(self) -> dict:
        job = self.backend.run(self.circuit, shots=1024)
        result = job.result()
        counts = result.get_counts(self.circuit)
        return counts


class QuantumTokenSearcher:
    def __init__(self, model, manager: QuantumChannelManager):
        self.model = model
        self.manager = manager

    def search_tokens(self, query: str) -> List[str]:
        binary_query = ''.join(['1' if char == 'a' else '0' for char in query.lower()])
        binary_query = binary_query.ljust(int(np.ceil(len(binary_query)/self.model.num_blocks)), '0')
        grover = GroverSearch(target_state=binary_query[:self.model.num_blocks])
        counts = grover.run()
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        top_states = [state for state, count in sorted_counts[:5]]
        token_indices = [int(state, 2) for state in top_states if int(state, 2) < self.model.vocab_size]
        tokens = [self.model.id_to_token.get(idx, "<UNK>") for idx in token_indices]
        return tokens


# --------------------------------------------------------------------
# Quantum Transformer Block (example usage)
# --------------------------------------------------------------------
class QuantumTransformerBlock:
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        sim_method: str = 'cpu',
        num_threads: int = 1,
        block_prefix: str = "block",
        enable_logging: bool = True,
        use_advanced_ansatz: bool = False,
        use_data_reuploading: bool = False,
        qc_manager: QuantumChannelManager = None,
        decoder: SubBitDecoder = None
    ):
        self.attn = QuantumAttentionLayer(
            embed_dim, 
            num_heads,
            sim_method=sim_method,
            num_threads=num_threads,
            prefix=f"{block_prefix}_attn",
            enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading
        )
        self.ffn = QuantumFeedForwardLayer(
            embed_dim, 
            hidden_dim,
            sim_method=sim_method,
            num_threads=num_threads,
            prefix=f"{block_prefix}_ffn",
            enable_logging=enable_logging,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading
        )
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qc_manager = qc_manager
        self.decoder = decoder
        self.token_searcher = QuantumTokenSearcher(model=None, manager=self.qc_manager)
        logging.info(f"Initialized QuantumTransformerBlock '{block_prefix}' with Qc integration.")

    def forward(self, x: np.ndarray, use_residual: bool = True) -> np.ndarray:
        x = x.astype(float).flatten()
        allocated_qcs = self.qc_manager.allocate_channels(self.num_heads)
        logging.debug(f"QuantumTransformerBlock: Allocated {len(allocated_qcs)} Qc's for attention.")

        for qc, value in zip(allocated_qcs, x):
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    scalar_value = float(value.item())
                else:
                    raise ValueError(f"Expected scalar value for encoding, got array size {value.size}")
            else:
                scalar_value = float(value)
            qc.encode(scalar_value)

        def attention_transform(circuit: QuantumCircuit, params: Optional[dict]) -> QuantumCircuit:
            theta = params.get('theta_rz', np.pi / 4)
            circuit.rz(float(theta), 0)
            return circuit

        attn_output = self.decoder.decode_and_transform(
            allocated_channels=allocated_qcs,
            transform_function=attention_transform,
            params={'theta_rz': np.pi / 4}
        )
        attn_output = float(attn_output[0])
        self.qc_manager.release_channels(allocated_qcs)

        if use_residual:
            x = normalize_vector(x + attn_output)
        else:
            x = attn_output

        allocated_qcs_ffn = self.qc_manager.allocate_channels(2)
        split_size = len(x) // 2
        values_to_encode = [float(x[i * split_size:(i + 1) * split_size].mean()) for i in range(2)]
        for qc, value in zip(allocated_qcs_ffn, values_to_encode):
            qc.encode(value)

        def ffn_transform(circuit: QuantumCircuit, params: Optional[dict]) -> QuantumCircuit:
            theta = params.get('theta_ry', np.pi / 6)
            circuit.ry(float(theta), 0)
            return circuit

        ffn_output = self.decoder.decode_and_transform(
            allocated_channels=allocated_qcs_ffn,
            transform_function=ffn_transform,
            params={'theta_ry': np.pi / 6}
        )
        ffn_output = float(ffn_output[0])
        self.qc_manager.release_channels(allocated_qcs_ffn)

        if use_residual:
            x = normalize_vector(x + ffn_output)
        else:
            x = ffn_output
        return x

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([self.attn.get_all_parameters(), self.ffn.get_all_parameters()])

    def set_all_parameters(self, params: np.ndarray):
        attn_size = len(self.attn.get_all_parameters())
        ffn_size = len(self.ffn.get_all_parameters())
        if params.shape[0] != attn_size + ffn_size:
            raise ValueError("Parameter mismatch in QuantumTransformerBlock.")
        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:])

    def to_dict(self) -> dict:
        return {
            "attn": self.attn.to_dict(),
            "ffn": self.ffn.to_dict()
        }

    def from_dict(self, d: dict):
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])


# --------------------------------------------------------------------
# QuantumLanguageModel
# --------------------------------------------------------------------
class QuantumLanguageModel:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        sim_method: str = 'cpu',
        num_threads: int = 1,
        enable_logging: bool = True,
        use_advanced_ansatz: bool = False,
        use_data_reuploading: bool = False,
        num_blocks: int = 1,
        use_context=False,
        use_positional_encoding=False,
        use_knowledge_embedding=False,
        knowledge_dim=0,
        manager: QuantumChannelManager = None,
        decoder: SubBitDecoder = None
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embeddings = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        self.blocks = []
        self.manager = manager if manager else QuantumChannelManager()
        if num_blocks > 1:
            for b in range(num_blocks):
                block_prefix = f"layer{b+1}"
                block = QuantumTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    sim_method=sim_method,
                    num_threads=num_threads,
                    block_prefix=block_prefix,
                    enable_logging=enable_logging,
                    use_advanced_ansatz=use_advanced_ansatz,
                    use_data_reuploading=use_data_reuploading,
                    qc_manager=self.manager,
                    decoder=decoder
                )
                self.blocks.append(block)
        else:
            self.attn = QuantumAttentionLayer(
                embed_dim, 
                num_heads,
                sim_method=sim_method,
                num_threads=num_threads,
                prefix="layer1_attn",
                enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading
            )
            self.ffn = QuantumFeedForwardLayer(
                embed_dim, 
                hidden_dim,
                sim_method=sim_method,
                num_threads=num_threads,
                prefix="layer1_ffn",
                enable_logging=enable_logging,
                use_advanced_ansatz=use_advanced_ansatz,
                use_data_reuploading=use_data_reuploading
            )
        self.W_proj = (np.random.randn(embed_dim, hidden_dim) * 0.01).astype(np.float32)
        self.W_out = (np.random.randn(vocab_size, embed_dim) * 0.01).astype(np.float32)
        self._initialize_quantum_params()
        self.num_blocks = num_blocks
        self.use_context = use_context
        self.use_positional_encoding = use_positional_encoding
        self.use_knowledge_embedding = use_knowledge_embedding
        self.knowledge_dim = knowledge_dim
        if self.use_context:
            self.context_module = QuantumContextModule()
        else:
            self.context_module = None
        if self.use_positional_encoding:
            self.pos_enc = QuantumPositionalEncoding()
        else:
            self.pos_enc = None
        if self.use_knowledge_embedding and knowledge_dim > 0:
            self.knowledge_module = QuantumKnowledgeEmbedding(knowledge_dim)
        else:
            self.knowledge_module = None
        self.qc_manager = self.manager
        self.decoder = decoder if decoder else SubBitDecoder(self.qc_manager)
        self.token_searcher = QuantumTokenSearcher(model=self, manager=self.qc_manager)

    def _initialize_quantum_params(self):
        scale = 0.1
        if self.blocks:
            for block in self.blocks:
                block.attn.query_params.set_values(np.random.randn(block.attn.query_params.size) * scale)
                block.attn.key_params.set_values(np.random.randn(block.attn.key_params.size) * scale)
                block.attn.value_params.set_values(np.random.randn(block.attn.value_params.size) * scale)
                block.attn.out_params.set_values(np.random.randn(block.attn.out_params.size) * scale)
                block.ffn.w1_params.set_values(np.random.randn(block.ffn.w1_params.size) * scale)
                block.ffn.w2_params.set_values(np.random.randn(block.ffn.w2_params.size) * scale)
        else:
            self.attn.query_params.set_values(np.random.randn(self.attn.query_params.size) * scale)
            self.attn.key_params.set_values(np.random.randn(self.attn.key_params.size) * scale)
            self.attn.value_params.set_values(np.random.randn(self.attn.value_params.size) * scale)
            self.attn.out_params.set_values(np.random.randn(self.attn.out_params.size) * scale)
            self.ffn.w1_params.set_values(np.random.randn(self.ffn.w1_params.size) * scale)
            self.ffn.w2_params.set_values(np.random.randn(self.ffn.w2_params.size) * scale)

    def forward(self, input_ids: List[int], use_residual: bool = True) -> np.ndarray:
        if not input_ids:
            raise ValueError("input_ids is empty.")
        for idx in input_ids:
            if idx < 0 or idx >= self.vocab_size:
                raise ValueError(f"Input id {idx} out of range.")
        x = self.embeddings[input_ids[-1]]
        if self.pos_enc:
            for i, token_id in enumerate(input_ids):
                x = self.pos_enc.apply_encoding(x, i)
            x = np.real(x)
        if self.knowledge_module:
            knowledge_state = self.knowledge_module.retrieve_knowledge_state(x)
            x = 0.5 * x + 0.5 * knowledge_state
        if self.blocks:
            for block in self.blocks:
                x = block.forward(x, use_residual=use_residual)
        else:
            attn_query = self.attn.forward(x, mode='query')
            attn_key = self.attn.forward(x, mode='key')
            attn_value = self.attn.forward(x, mode='value')
            attn_out = self.attn.forward(x, mode='out')
            attn_output = attn_query + attn_key + attn_value + attn_out
            if use_residual:
                x = normalize_vector(x + attn_output)
            else:
                x = attn_output
            ffn_w1 = self.ffn.forward(x, layer='w1')
            ffn_w2 = self.ffn.forward(ffn_w1, layer='w2')
            if use_residual:
                x = normalize_vector(x + ffn_w2)
            else:
                x = ffn_w2
        logits = self.W_out @ x
        if self.context_module and logits is not None:
            self.context_module.store_state(logits)
        return logits

    def get_all_parameters(self) -> np.ndarray:
        if self.blocks:
            block_params = []
            for block in self.blocks:
                block_params.append(block.get_all_parameters())
            stacked_block_params = np.concatenate(block_params)
        else:
            stacked_block_params = np.concatenate([
                self.attn.get_all_parameters(),
                self.ffn.get_all_parameters()
            ])
        return np.concatenate([
            stacked_block_params,
            self.W_proj.flatten(),
            self.W_out.flatten()
        ])

    def set_all_parameters(self, params: np.ndarray):
        if self.blocks:
            total_block_params = 0
            block_sizes = []
            for block in self.blocks:
                size_block = len(block.get_all_parameters())
                block_sizes.append(size_block)
                total_block_params += size_block
            proj_size = (self.embed_dim * self.hidden_dim)
            out_size = (self.vocab_size * self.embed_dim)
            expected = total_block_params + proj_size + out_size
            if params.shape[0] != expected:
                raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")
            offset = 0
            for i, block in enumerate(self.blocks):
                block_param_size = block_sizes[i]
                block_params = params[offset: offset + block_param_size]
                block.set_all_parameters(block_params)
                offset += block_param_size
            self.W_proj = params[offset: offset + proj_size].reshape(self.embed_dim, self.hidden_dim)
            offset += proj_size
            self.W_out = params[offset: offset + out_size].reshape(self.vocab_size, self.embed_dim)
        else:
            attn_size = (
                self.attn.query_params.size +
                self.attn.key_params.size +
                self.attn.value_params.size +
                self.attn.out_params.size
            )
            ffn_size = self.ffn.w1_params.size + self.ffn.w2_params.size
            proj_size = self.embed_dim * self.hidden_dim
            out_size = self.vocab_size * self.embed_dim
            expected = attn_size + ffn_size + proj_size + out_size
            if params.shape[0] != expected:
                raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")
            self.attn.set_all_parameters(params[:attn_size])
            self.ffn.set_all_parameters(params[attn_size: attn_size + ffn_size])
            self.W_proj = params[attn_size + ffn_size: attn_size + ffn_size + proj_size].reshape(self.embed_dim, self.hidden_dim)
            self.W_out = params[attn_size + ffn_size + proj_size:].reshape(self.vocab_size, self.embed_dim)

    def search_related_tokens(self, query: str) -> List[str]:
        return self.token_searcher.search_tokens(query)

    def to_dict(self) -> dict:
        model_dict = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings.tolist(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "version": "4.0",
            "num_blocks": self.num_blocks,
            "use_context": self.use_context,
            "use_positional_encoding": self.use_positional_encoding,
            "use_knowledge_embedding": self.use_knowledge_embedding,
            "knowledge_dim": self.knowledge_dim
        }
        if self.blocks:
            block_dicts = []
            for block in self.blocks:
                block_dicts.append(block.to_dict())
            model_dict["blocks"] = block_dicts
        else:
            model_dict["attn"] = self.attn.to_dict()
            model_dict["ffn"] = self.ffn.to_dict()
        return model_dict

    def from_dict(self, d: dict):
        if (
            d["vocab_size"] != self.vocab_size or
            d["embed_dim"] != self.embed_dim or
            d["num_heads"] != self.num_heads or
            d["hidden_dim"] != self.hidden_dim
        ):
            raise ValueError("Model config mismatch.")
        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)
        self.num_blocks = d.get("num_blocks", 1)
        self.use_context = d.get("use_context", False)
        self.use_positional_encoding = d.get("use_positional_encoding", False)
        self.use_knowledge_embedding = d.get("use_knowledge_embedding", False)
        self.knowledge_dim = d.get("knowledge_dim", 0)
        if self.use_context:
            self.context_module = QuantumContextModule()
        else:
            self.context_module = None
        if self.use_positional_encoding:
            self.pos_enc = QuantumPositionalEncoding()
        else:
            self.pos_enc = None
        if self.use_knowledge_embedding and self.knowledge_dim > 0:
            self.knowledge_module = QuantumKnowledgeEmbedding(self.knowledge_dim)
        else:
            self.knowledge_module = None
        if self.num_blocks > 1 and "blocks" in d:
            self.blocks = []
            for i, block_info in enumerate(d["blocks"]):
                block_prefix = f"layer{i+1}"
                new_block = QuantumTransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    sim_method='cpu',
                    num_threads=1,
                    block_prefix=block_prefix,
                    enable_logging=False,
                    use_advanced_ansatz=False,
                    use_data_reuploading=False,
                    qc_manager=self.manager,
                    decoder=self.decoder
                )
                new_block.from_dict(block_info)
                self.blocks.append(new_block)
        else:
            self.attn.from_dict(d["attn"])
            self.ffn.from_dict(d["ffn"])

    def save_model(self, save_path: str):
        model_dict = self.to_dict()
        with open(save_path, 'w') as f:
            json.dump(model_dict, f)
        logging.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")
        with open(load_path, 'r') as f:
            model_dict = json.load(f)
        if "version" not in model_dict or model_dict["version"] != "4.0":
            raise ValueError("Unsupported model version.")
        self.from_dict(model_dict)
        logging.info(f"Model loaded from {load_path}")

    def shift_parameter(self, param_index: int, shift: float):
        shifted_params = self.get_all_parameters()
        shifted_params[param_index] += shift
        self.set_all_parameters(shifted_params)

    def unshift_parameter(self, param_index: int, shift: float):
        self.shift_parameter(param_index, -shift)


class QuantumContextModule:
    def __init__(self):
        self.conversation_states = []

    def store_state(self, quantum_state: np.ndarray):
        self.conversation_states.append(quantum_state)

    def clear_states(self):
        self.conversation_states = []

    def get_aggregated_context(self) -> np.ndarray:
        if not self.conversation_states:
            return None
        stacked = np.vstack(self.conversation_states)
        avg_state = np.mean(stacked, axis=0)
        return normalize_vector(avg_state)


class QuantumPositionalEncoding:
    def apply_encoding(self, input_state: np.ndarray, position: int) -> np.ndarray:
        alpha = 0.05
        phase_factor = np.exp(1j * alpha * position)
        encoded_state = input_state * phase_factor
        return encoded_state


class QuantumKnowledgeEmbedding:
    def __init__(self, knowledge_dim: int):
        self.knowledge_dim = knowledge_dim
        self.knowledge_matrix = np.random.randn(knowledge_dim, knowledge_dim)

    def retrieve_knowledge_state(self, query: np.ndarray) -> np.ndarray:
        retrieval = self.knowledge_matrix @ query
        return normalize_vector(retrieval)


def quantum_data_augmentation(input_data: np.ndarray) -> np.ndarray:
    noise_level = 0.001
    noisy_data = input_data + noise_level * np.random.randn(*input_data.shape)
    return normalize_vector(noisy_data)


def quantum_batch_shift_training(model, X, Y, batch_size=32, lr=0.001):
    pass


class QuantumParameterStore:
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
    def __init__(
        self,
        sim_method: str = 'cpu',
        num_threads: int = 1,
        enable_logging: bool = True,
        use_advanced_ansatz: bool = False,
        use_data_reuploading: bool = False
    ):
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.use_advanced_ansatz = use_advanced_ansatz
        self.use_data_reuploading = use_data_reuploading
        self.backend = self.initialize_simulator()

    def initialize_simulator(self):
        if self.sim_method == 'gpu':
            backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using GPU.")
        elif self.sim_method == 'both':
            backend = AerSimulator(method='statevector', device='GPU', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using Both CPU and GPU.")
        elif self.sim_method == 'simulation':
            backend = None
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using pure simulation mode.")
        else:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using CPU.")
        return backend

    def build_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        if self.use_advanced_ansatz:
            circuit = self.build_advanced_circuit(input_vector, param_store)
        else:
            circuit = self.build_simple_circuit(input_vector, param_store)
        return circuit

    def build_simple_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
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
                circuit.cx(i, i + 1)
        for i in range(qubits_needed):
            theta = param_store.values[num_layers * qubits_needed + i]
            circuit.ry(theta, i)
        circuit.save_statevector()
        return circuit

    def build_advanced_circuit(self, input_vector: np.ndarray, param_store: QuantumParameterStore) -> QuantumCircuit:
        qubits_needed = max(1, int(np.ceil(np.log2(len(input_vector)))))
        circuit = QuantumCircuit(qubits_needed)
        state_prep_vec = np.zeros(2**qubits_needed, dtype=complex)
        state_prep_vec[:len(input_vector)] = input_vector.astype(complex)
        state_prep_vec = normalize_vector(state_prep_vec)
        circuit.initialize(state_prep_vec, qubits=range(qubits_needed))
        layers = 2
        offset = 0
        for _ in range(layers):
            for i in range(qubits_needed):
                theta_ry = float(param_store.values[offset])
                offset += 1
                circuit.ry(theta_ry, i)

                theta_rz = 0
                if offset < param_store.size:
                    theta_rz = float(param_store.values[offset])
                    offset += 1
                    circuit.rz(theta_rz, i)

                if self.use_data_reuploading:
                    scaled_angle = float(input_vector[i % len(input_vector)]) * 0.1
                    circuit.rx(scaled_angle, i)
            for i in range(qubits_needed):
                next_qubit = (i + 1) % qubits_needed
                circuit.cx(i, next_qubit)
        circuit.save_statevector()
        return circuit

    def simulate(self, circuit: QuantumCircuit) -> np.ndarray:
        if self.sim_method == 'simulation' or self.backend is None:
            data = circuit.data
            qubits_needed = circuit.num_qubits
            if len(data) > 0 and data[0].operation.name == 'initialize':
                init_vec = data[0].operation.params[0]
                return init_vec
            else:
                return np.zeros(2**qubits_needed, dtype=complex)
        else:
            job = self.backend.run(circuit, shots=1)
            result = job.result()
            final_state = result.get_statevector(circuit)
            return final_state.data


class AdamOptimizer:
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


class AdvancedQuantumOptimizer:
    def __init__(self, parameters: np.ndarray, lr: float = 0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self, gradients: np.ndarray):
        self.parameters -= self.lr * gradients
        return self.parameters


def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randint(4, vocab_size, size=(num_samples,))
    Y = np.random.randint(4, vocab_size, size=(num_samples,))
    return X, Y


def load_real_dataset(file_path: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = word_tokenize(text.lower())
    from collections import defaultdict
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in sorted_tokens:
        if len(token_to_id) >= vocab_size:
            break
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    X, Y_ids = [], []
    for i in range(len(tokens) - 1):
        current_token = tokens[i]
        next_token = tokens[i + 1]
        X.append(token_to_id.get(current_token, token_to_id["<UNK>"]))
        Y_ids.append(token_to_id.get(next_token, token_to_id["<UNK>"]))
    Y = np.array(Y_ids, dtype=np.int32)
    return np.array(X), Y, token_to_id


def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    logits = logits - np.max(logits)
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    softmax = np.clip(softmax, 1e-12, 1.0)
    return -np.log(softmax[target])


def perplexity(logits: np.ndarray, target: int) -> float:
    ce_loss = cross_entropy_loss(logits, target)
    return np.exp(ce_loss)


def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    from collections import Counter
    import math
    weights = [1.0 / max_n] * max_n
    reference_counts = [
        Counter([tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)])
        for n in range(1, max_n + 1)
    ]
    hypothesis_counts = [
        Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)])
        for n in range(1, max_n + 1)
    ]
    precisions = []
    for ref_count, hyp_count in zip(reference_counts, hypothesis_counts):
        overlap = hyp_count & ref_count
        precision = sum(overlap.values()) / max(sum(hyp_count.values()), 1e-12)
        precisions.append(precision)
    ref_length = len(reference)
    hyp_length = len(hypothesis)
    if hyp_length == 0:
        bp = 0
    elif hyp_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / hyp_length)
    if min(precisions) > 0:
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
    else:
        geo_mean = 0
    return bp * geo_mean


def compute_gradient_for_parameter(args):
    (
        vocab_size, 
        embed_dim, 
        num_heads, 
        hidden_dim,
        sim_method, 
        num_threads, 
        X, 
        Y, 
        original_params, 
        i,
        use_advanced_ansatz, 
        use_data_reuploading, 
        num_blocks,
        use_context,
        use_positional_encoding,
        use_knowledge_embedding,
        knowledge_dim
    ) = args
    try:
        manager = QuantumChannelManager()
        required_channels = num_blocks * (num_heads + 2)
        manager.create_channels(num_channels=required_channels)
        decoder = SubBitDecoder(manager=manager)
        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            sim_method=sim_method,
            num_threads=num_threads,
            enable_logging=False,
            use_advanced_ansatz=use_advanced_ansatz,
            use_data_reuploading=use_data_reuploading,
            num_blocks=num_blocks,
            use_context=use_context,
            use_positional_encoding=use_positional_encoding,
            use_knowledge_embedding=use_knowledge_embedding,
            knowledge_dim=knowledge_dim,
            manager=manager,
            decoder=decoder
        )
        model.set_all_parameters(original_params)
        shift = np.pi / 2
        model.shift_parameter(i, shift)
        loss_plus = np.mean([
            cross_entropy_loss(model.forward([x], use_residual=True), y) 
            for x, y in zip(X, Y)
        ])
        model.unshift_parameter(i, shift)
        loss_minus = np.mean([
            cross_entropy_loss(model.forward([x], use_residual=True), y) 
            for x, y in zip(X, Y)
        ])
        gradient = (loss_plus - loss_minus) / 2.0
        return i, gradient
    except Exception:
        traceback.print_exc()
        return i, 0.0


def compute_gradients_parallel(
    model: QuantumLanguageModel,
    X, Y,
    num_processes: int = 1,
    progress_callback=None,
    batch_shifts: bool = False
) -> np.ndarray:
    gradients = np.zeros_like(model.get_all_parameters())
    original_params = model.get_all_parameters().copy()
    total_params = len(original_params)
    block_size = 100
    args_list = []
    sim_method_used = model.blocks[0].attn.sim_method if model.blocks else model.attn.sim_method
    num_threads_used = model.blocks[0].attn.num_threads if model.blocks else model.attn.num_threads
    use_advanced_ansatz_used = model.blocks[0].attn.use_advanced_ansatz if model.blocks else model.attn.use_advanced_ansatz
    use_data_reuploading_used = model.blocks[0].attn.use_data_reuploading if model.blocks else model.attn.use_data_reuploading
    for i in range(total_params):
        args_list.append((
            model.vocab_size,
            model.embed_dim,
            model.num_heads,
            model.hidden_dim,
            sim_method_used,
            num_threads_used,
            X,
            Y,
            original_params,
            i,
            use_advanced_ansatz_used,
            use_data_reuploading_used,
            model.num_blocks,
            model.use_context,
            model.use_positional_encoding,
            model.use_knowledge_embedding,
            model.knowledge_dim
        ))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(compute_gradient_for_parameter, args): args[-3] for args in args_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, gradient = future.result()
            gradients[i] = gradient
            completed += 1
            if progress_callback and (completed % block_size == 0 or completed == total_params):
                progress_callback(completed, total_params, i, gradient)
    return gradients


def train_model(
    model: QuantumLanguageModel,
    X, Y,
    epochs: int = 10,
    lr: float = 0.001,
    num_threads: int = 1,
    log_queue: queue.Queue = None,
    stop_flag=None,
    time_lock: threading.Lock = None,
    time_data=None,
    optimizer=None,
    use_data_augmentation=False,
    use_batch_shift=False
):
    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs
    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break
        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")
        epoch_start_time = time.time()
        def progress_callback(completed, total, param_index, gradient):
            if log_queue:
                log_queue.put(f"PROGRESS:gradient,{completed},{total}\n")
                avg_grad = np.mean(np.abs(gradient))
                log_queue.put(f"INFO:Parameter {param_index} Gradient Magnitude: {avg_grad:.6f}\n")
        if use_data_augmentation:
            augmented_X = []
            for x_val in X:
                embedded_x = model.embeddings[x_val]
                new_x = quantum_data_augmentation(embedded_x)
                augmented_X.append(new_x)
            # demonstration only
        gradients = compute_gradients_parallel(
            model,
            X,
            Y,
            num_processes=num_threads,
            progress_callback=progress_callback,
            batch_shifts=use_batch_shift
        )
        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            params = model.get_all_parameters()
            params -= lr * gradients
            model.set_all_parameters(params)
        total_loss = np.mean([cross_entropy_loss(model.forward([x], use_residual=True), y) for x, y in zip(X, Y)])
        total_perplexity = np.mean([perplexity(model.forward([x], use_residual=True), y) for x, y in zip(X, Y)])
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if log_queue:
            log_queue.put(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, "
                f"Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n"
            )
        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0
        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"PROGRESS:epoch,{epoch_progress}\n")
    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_inference(
    model: QuantumLanguageModel,
    input_sequence: List[int],
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    max_length: int = 50,
    temperature: float = 1.0,
    log_callback=None
):
    generated = input_sequence.copy()
    for _ in range(max_length):
        logits = model.forward([generated[-1]], use_residual=True)
        probabilities = softmax(logits / temperature)
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        generated.append(chosen_index)
        if chosen_index == token_to_id.get("<END>", chosen_index):
            break
    generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
    response = " ".join(generated_tokens)
    if log_callback:
        log_callback(f"Generated Response:\n{response}\n\n")
    return generated_tokens, response


def get_gpu_usage() -> Optional[str]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip():
            usage_str = result.stdout.strip().split('\n')[0]
            return f"{usage_str}%"
        else:
            return "N/A"
    except FileNotFoundError:
        return "N/A"
    except Exception:
        return "N/A"


# --------------------------------------------------------------------
# GUI (QELM)
# --------------------------------------------------------------------
class QELM_GUI:
    def __init__(self, master):
        try:
            self.master = master
            master.title("QELM Trainer - V-Q")
            master.geometry("1440x900")
            master.resizable(False, False)
            self.vocab_size = 100
            self.embed_dim = 256
            self.num_heads = 8
            self.hidden_dim = 512
            self.sim_method = 'cpu'
            self.num_threads = min(8, multiprocessing.cpu_count())
            self.use_advanced_ansatz = False
            self.use_data_reuploading = False
            self.num_blocks = 1
            # decimal_precision is used for channel encoding
            self.decimal_precision = 4

            self.model = QuantumLanguageModel(
                self.vocab_size,
                self.embed_dim,
                self.num_heads,
                self.hidden_dim,
                sim_method=self.sim_method,
                num_threads=self.num_threads,
                enable_logging=True,
                use_advanced_ansatz=self.use_advanced_ansatz,
                use_data_reuploading=self.use_data_reuploading,
                num_blocks=self.num_blocks
            )
            self.token_to_id = {}
            self.id_to_token = {}
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=0.001)
            self.stop_flag = threading.Event()
            self.time_data = {'start_time': None, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
            self.time_lock = threading.Lock()
            if psutil:
                self.process = psutil.Process(os.getpid())
                self.process.cpu_percent(interval=None)
            else:
                self.process = None
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
            style.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
            style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
            style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])
            self.create_widgets()
            self.update_resource_usage()
            self.update_time_label()
            self.log_queue = queue.Queue()
            self.master.after(100, self.process_log_queue)
            self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.error_log_path = None
            self.setup_error_logging()
        except Exception as e:
            error_trace = traceback.format_exc()
            logging.critical(f"GUI Initialization error:\n{error_trace}")
            messagebox.showerror("Initialization Error", f"An error occurred during GUI initialization:\n{e}\n\nCheck the log file for more details.")
            sys.exit(1)

    def setup_error_logging(self):
        try:
            self.error_logger = logging.getLogger('error_logger')
            self.error_logger.setLevel(logging.ERROR)
            if not self.error_logger.handlers:
                self.error_log_handler = logging.FileHandler('error.log')
                self.error_log_handler.setLevel(logging.ERROR)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                self.error_log_handler.setFormatter(formatter)
                self.error_logger.addHandler(self.error_log_handler)
        except Exception as e:
            logging.error(f"Failed to setup error logging: {e}")

    def on_closing(self):
        self.stop_flag.set()
        self.master.destroy()

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

        # Train Model Tab
        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)
        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        select_dataset_btn = ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset)
        select_dataset_btn.pack(side='right', padx=10, pady=10)

        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Model Parameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)
        hyperparams_left = ttk.Frame(hyperparams_frame)
        hyperparams_left.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(hyperparams_left, text="Vocabulary Size:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.vocab_size_entry = ttk.Entry(hyperparams_left, width=15, style="Custom.TEntry")
        self.vocab_size_entry.insert(0, str(self.vocab_size))
        self.vocab_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_left, text="Embedding Dimension:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.embed_dim_entry = ttk.Entry(hyperparams_left, width=15, style="Custom.TEntry")
        self.embed_dim_entry.insert(0, str(self.embed_dim))
        self.embed_dim_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_left, text="Number of Attention Heads:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.num_heads_entry = ttk.Entry(hyperparams_left, width=15, style="Custom.TEntry")
        self.num_heads_entry.insert(0, str(self.num_heads))
        self.num_heads_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        hyperparams_right = ttk.Frame(hyperparams_frame)
        hyperparams_right.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(hyperparams_right, text="Hidden Dimension:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.hidden_dim_entry = ttk.Entry(hyperparams_right, width=15, style="Custom.TEntry")
        self.hidden_dim_entry.insert(0, str(self.hidden_dim))
        self.hidden_dim_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_right, text="Learning Rate:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_right, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_right, text="Epochs:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.epochs_entry = ttk.Entry(hyperparams_right, width=15, style="Custom.TEntry")
        self.epochs_entry.insert(0, "2")
        self.epochs_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        sim_settings_frame = ttk.LabelFrame(self.tab_train, text="Simulation Settings")
        sim_settings_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(sim_settings_frame, text="Simulation Method:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.sim_method_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(sim_settings_frame, text='CPU', variable=self.sim_method_var, value='cpu', command=self.update_threads_based_on_method)
        gpu_radio = ttk.Radiobutton(sim_settings_frame, text='GPU', variable=self.sim_method_var, value='gpu', command=self.update_threads_based_on_method)
        both_radio = ttk.Radiobutton(sim_settings_frame, text='Both CPU/GPU', variable=self.sim_method_var, value='both', command=self.update_threads_based_on_method)
        simulation_radio = ttk.Radiobutton(sim_settings_frame, text='Simulation', variable=self.sim_method_var, value='simulation', command=self.update_threads_based_on_method)
        cpu_radio.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        gpu_radio.grid(row=0, column=2, padx=10, pady=10, sticky='w')
        both_radio.grid(row=0, column=3, padx=10, pady=10, sticky='w')
        simulation_radio.grid(row=0, column=4, padx=10, pady=10, sticky='w')

        ttk.Label(sim_settings_frame, text="Number of Threads:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(sim_settings_frame, from_=1, to=multiprocessing.cpu_count(), textvariable=self.num_threads_var, width=5)
        self.num_threads_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=1, column=2, padx=10, pady=10, sticky='w')

        adv_settings_frame = ttk.LabelFrame(self.tab_train, text="Advanced Quantum Settings")
        adv_settings_frame.pack(fill='x', padx=10, pady=10)
        self.use_advanced_ansatz_var = tk.BooleanVar(value=False)
        self.use_data_reuploading_var = tk.BooleanVar(value=False)
        self.num_blocks_var = tk.IntVar(value=1)

        ttk.Checkbutton(adv_settings_frame, text='Use Advanced Ansatz', variable=self.use_advanced_ansatz_var).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        ttk.Checkbutton(adv_settings_frame, text='Use Data Reuploading', variable=self.use_data_reuploading_var).grid(row=0, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(adv_settings_frame, text="Number of Blocks:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.blocks_spinbox = ttk.Spinbox(adv_settings_frame, from_=1, to=10, textvariable=self.num_blocks_var, width=5)
        self.blocks_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # decimal precision spinbox
        ttk.Label(adv_settings_frame, text="Decimal Precision:").grid(row=1, column=2, padx=10, pady=10, sticky='e')
        self.decimal_precision_var = tk.IntVar(value=self.decimal_precision)
        self.decimal_precision_spinbox = ttk.Spinbox(adv_settings_frame, from_=0, to=10, textvariable=self.decimal_precision_var, width=5)
        self.decimal_precision_spinbox.grid(row=1, column=3, padx=10, pady=10, sticky='w')

        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)

        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)
        stop_button = ttk.Button(train_controls_frame, text="Stop (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=10, pady=10)
        hard_stop_button = ttk.Button(train_controls_frame, text="Hard Stop (Immediate)", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=10, pady=10)
        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)
        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)

        progress_bars_frame = ttk.Frame(self.tab_train)
        progress_bars_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(progress_bars_frame, text="Training Progress:").pack(anchor='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.epoch_progress.pack(fill='x', padx=10, pady=5)

        ttk.Label(progress_bars_frame, text="Gradient Computation Progress:").pack(anchor='w', padx=10, pady=5)
        self.gradient_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.gradient_progress.pack(fill='x', padx=10, pady=5)

        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)

        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Evaluation Metrics")
        eval_metrics_frame.pack(fill='x', padx=10, pady=10)
        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.pack(anchor='w', padx=10, pady=5)
        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU Score: N/A")
        self.bleu_label.pack(anchor='w', padx=10, pady=5)

        # Inference Tab
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
        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)

        # Manage Token Mappings Tab
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)
        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)
        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10), bg="#2C3E50", fg="white", insertbackground="white")
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)

        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)
        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)
        self.gpu_label = ttk.Label(usage_frame, text="GPU: N/A")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)
        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating...")
        self.time_label.pack(anchor='w', padx=10, pady=5)

        error_log_frame = ttk.LabelFrame(right_frame, text="Error Log Configuration")
        error_log_frame.pack(fill='x', padx=5, pady=20, side='bottom')
        select_error_log_btn = ttk.Button(error_log_frame, text="Select Error Log Save Location", command=self.select_error_log)
        select_error_log_btn.pack(side='top', padx=10, pady=10)
        self.error_log_path_var = tk.StringVar(value="No error log selected.")
        ttk.Label(error_log_frame, textvariable=self.error_log_path_var).pack(side='top', padx=10, pady=5)

    def select_error_log(self):
        try:
            file_path = filedialog.asksaveasfilename(title="Select Error Log Save Location", defaultextension=".log", filetypes=[("Log Files", "*.log"), ("All Files", "*.*")])
            if file_path:
                self.error_log_path = file_path
                self.error_log_path_var.set(file_path)
                for handler in self.error_logger.handlers[:]:
                    self.error_logger.removeHandler(handler)
                self.error_log_handler = logging.FileHandler(self.error_log_path)
                self.error_log_handler.setLevel(logging.ERROR)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                self.error_log_handler.setFormatter(formatter)
                self.error_logger.addHandler(self.error_log_handler)
                self.log_train(f"Error log will be saved to {self.error_log_path}\n")
        except Exception as e:
            err_msg = f"Error selecting error log save location:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Error Log Save Error", err_msg)

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message.startswith("PROGRESS:gradient"):
                    try:
                        _, gradient_info = message.split(":", 1)
                        _, completed, total = gradient_info.strip().split(",")
                        completed = int(completed)
                        total = int(total)
                        progress_percentage = (completed / total) * 100
                        self.gradient_progress['value'] = progress_percentage
                        self.gradient_progress.update()
                    except Exception:
                        self.train_log.config(state='normal')
                        self.train_log.insert(tk.END, message)
                        self.train_log.see(tk.END)
                        self.train_log.config(state='disabled')
                elif message.startswith("PROGRESS:epoch"):
                    try:
                        _, epoch_info = message.split(":", 1)
                        _, epoch_percentage = epoch_info.strip().split(",")
                        epoch_percentage = float(epoch_percentage)
                        self.epoch_progress['value'] = epoch_percentage
                        self.epoch_progress.update()
                    except Exception:
                        self.train_log.config(state='normal')
                        self.train_log.insert(tk.END, message)
                        self.train_log.see(tk.END)
                        self.train_log.config(state='disabled')
                elif message.startswith("INFO:"):
                    info_message = message.replace("INFO:", "")
                    self.train_log.config(state='normal')
                    self.train_log.insert(tk.END, f"{info_message}\n")
                    self.train_log.see(tk.END)
                    self.train_log.config(state='disabled')
                else:
                    self.train_log.config(state='normal')
                    self.train_log.insert(tk.END, message)
                    self.train_log.see(tk.END)
                    self.train_log.config(state='disabled')
        except Exception as e:
            logging.error(f"Error processing log queue: {e}")
            self.error_logger.error(f"Error processing log queue: {e}")
        finally:
            if not self.stop_flag.is_set():
                self.master.after(100, self.process_log_queue)

    def update_threads_based_on_method(self):
        method = self.sim_method_var.get()
        max_threads = multiprocessing.cpu_count()
        self.num_threads_spinbox.config(to=max_threads)
        if self.num_threads_var.get() > max_threads:
            self.num_threads_var.set(max_threads)

    def log_train(self, message: str):
        if hasattr(self, 'log_queue'):
            self.log_queue.put(message)
        if "error" in message.lower():
            self.error_logger.error(message)

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
                self.token_to_id = {}
                self.id_to_token = {}
                self.log_train(f"Selected Dataset: {file_path}\n")
        except Exception as e:
            err_msg = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Dataset Selection Error", err_msg)

    def train_model(self):
        try:
            vocab_size = int(self.vocab_size_entry.get())
            embed_dim = int(self.embed_dim_entry.get())
            num_heads = int(self.num_heads_entry.get())
            hidden_dim = int(self.hidden_dim_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            num_blocks = int(self.num_blocks_var.get())
            if vocab_size <= 0 or embed_dim <= 0 or num_heads <= 0 or hidden_dim <= 0 or lr <= 0 or epochs <= 0 or num_blocks <= 0:
                raise ValueError
            if embed_dim % num_heads != 0:
                messagebox.showerror("Invalid Input", "Embedding Dimension must be divisible by Number of Attention Heads.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive integers for model parameters and epochs, and a positive float for learning rate.")
            return

        sim_method = self.sim_method_var.get()
        num_threads = self.num_threads_var.get()
        max_threads = multiprocessing.cpu_count()
        if num_threads > max_threads:
            messagebox.showwarning("Thread Limit", f"Resetting threads to max {max_threads}")
            num_threads = max_threads
            self.num_threads_var.set(num_threads)
        self.use_advanced_ansatz = self.use_advanced_ansatz_var.get()
        self.use_data_reuploading = self.use_data_reuploading_var.get()
        self.num_blocks = num_blocks
        self.decimal_precision = self.decimal_precision_var.get()

        use_exponential_tokenizer = False

        if hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                if use_exponential_tokenizer:
                    X, Y, token_map, id_map = load_dataset_with_exponential_tokenizer(dataset_path, vocab_size)
                    self.X = X
                    self.Y = Y
                    self.token_to_id = token_map
                    self.id_to_token = id_map
                    self.log_train(f"Loaded dataset with exponential subword tokenizer from {dataset_path}\n")
                else:
                    X, Y, token_to_id = load_real_dataset(dataset_path, vocab_size)
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
            X, Y = create_synthetic_dataset(vocab_size, num_samples=500)
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")
            self.token_to_id = {f"<TOKEN_{i}>": i for i in range(vocab_size)}
            self.id_to_token = {i: f"<TOKEN_{i}>" for i in range(vocab_size)}

        try:
            manager = self.model.qc_manager
            decoder = self.model.decoder
            required_channels = self.num_blocks * (self.num_heads + 2)
            current_channels = len(manager.channels)
            if current_channels < required_channels:
                additional_channels = required_channels - current_channels
                manager.create_channels(num_channels=additional_channels, decimal_precision=self.decimal_precision)
                self.log_train(f"Created additional {additional_channels} Quantum Channels to meet the requirement.\n")
            self.model = QuantumLanguageModel(
                vocab_size,
                embed_dim,
                num_heads,
                hidden_dim,
                sim_method=sim_method,
                num_threads=num_threads,
                enable_logging=True,
                use_advanced_ansatz=self.use_advanced_ansatz,
                use_data_reuploading=self.use_data_reuploading,
                num_blocks=self.num_blocks,
                use_context=self.model.use_context,
                use_positional_encoding=self.model.use_positional_encoding,
                use_knowledge_embedding=self.model.use_knowledge_embedding,
                knowledge_dim=self.model.knowledge_dim,
                manager=manager,
                decoder=decoder
            )
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=lr)
            self.log_train("Model re-initialized with new parameters.\n")
        except Exception as e:
            err_msg = f"Failed to initialize model with new parameters:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Model Initialization Error", err_msg)
            return

        if sim_method == 'both':
            try:
                if not self.model.blocks:
                    self.model.attn.sim_method = 'cpu'
                    self.model.ffn.sim_method = 'cpu'
                    self.model.attn.backend = self.model.attn.initialize_simulator()
                    self.model.ffn.backend = self.model.ffn.initialize_simulator()
                else:
                    for block in self.model.blocks:
                        block.attn.sim_method = 'cpu'
                        block.ffn.sim_method = 'cpu'
                        block.attn.backend = block.attn.initialize_simulator()
                        block.ffn.backend = block.ffn.initialize_simulator()
                self.log_train("Initialized CPU-based simulator. GPU usage would be toggled in an actual environment.\n")
            except Exception as e:
                err_msg = f"Failed to initialize CPU/GPU simulators:\n{traceback.format_exc()}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("GPU Initialization Error", err_msg)
                return
        elif sim_method == 'simulation':
            self.log_train("Simulation mode selected. No real quantum circuit execution.\n")
        else:
            if not self.model.blocks:
                self.model.attn.sim_method = sim_method
                self.model.ffn.sim_method = sim_method
                self.model.attn.backend = self.model.attn.initialize_simulator()
                self.model.ffn.backend = self.model.ffn.initialize_simulator()
            else:
                for block in self.model.blocks:
                    block.attn.sim_method = sim_method
                    block.ffn.sim_method = sim_method
                    block.attn.backend = block.attn.initialize_simulator()
                    block.ffn.backend = block.ffn.initialize_simulator()

        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.infer_button.config(state='disabled')
        self.stop_flag.clear()
        self.epoch_progress['value'] = 0
        self.gradient_progress['value'] = 0
        self.train_log.config(state='normal')
        self.train_log.delete('1.0', tk.END)
        self.train_log.config(state='disabled')
        self.log_train("Starting training...\n")
        with self.time_lock:
            self.time_data['start_time'] = time.time()
            self.time_data['epochs_done'] = 0
            self.time_data['epochs'] = epochs
            self.time_data['remaining'] = 0
        training_thread = threading.Thread(target=self.training_process, args=(epochs, num_threads), daemon=True)
        training_thread.start()

    def training_process(self, epochs: int, num_threads: int):
        try:
            self.log_train("Training thread started.\n")
            train_model(
                self.model,
                self.X,
                self.Y,
                epochs=epochs,
                lr=self.optimizer.lr,
                num_threads=num_threads,
                log_queue=self.log_queue,
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
            self.error_logger.error(err_msg)
            messagebox.showerror("Training Error", err_msg)
        finally:
            with self.time_lock:
                self.time_data['start_time'] = None
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.infer_button.config(state='normal')
            if not self.stop_flag.is_set():
                self.epoch_progress['value'] = 100
                self.gradient_progress['value'] = 100
            self.evaluate_model()

    def stop_training(self):
        self.stop_flag.set()
        self.log_train("Stop signal sent. Will stop after current epoch.\n")

    def hard_stop(self):
        self.log_train("Hard stop invoked. Terminating immediately.\n")
        os._exit(1)

    def save_model(self):
        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm", filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if save_path:
                self.model.save_model(save_path)
                if len(self.token_to_id) != self.model.vocab_size:
                    raise ValueError(f"Token mapping size {len(self.token_to_id)} does not match model's vocab_size {self.model.vocab_size}.")
                if self.token_to_id:
                    base, ext = os.path.splitext(save_path)
                    token_map_path = f"{base}_token_map.json"
                    with open(token_map_path, 'w') as f:
                        json.dump(self.token_to_id, f, indent=4)
                    self.log_train(f"Token mappings saved to {token_map_path}\n")
                messagebox.showinfo("Model Saved", f"Model saved to {save_path}")
        except Exception as e:
            err_msg = f"Save model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            self.error_logger.error(err_msg)
            messagebox.showerror("Save Error", err_msg)

    def load_model(self):
        try:
            load_path = filedialog.askopenfilename(title="Load Model", filetypes=[("QELM Files", "*.qelm"), ("All Files", "*.*")])
            if load_path:
                self.model.load_model(load_path)
                base, ext = os.path.splitext(load_path)
                token_map_path = f"{base}_token_map.json"
                try:
                    with open(token_map_path, 'r') as f:
                        self.token_to_id = json.load(f)
                    self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                    if len(self.token_to_id) != self.model.vocab_size:
                        raise ValueError(f"Token mapping size {len(self.token_to_id)} does not match model's vocab_size {self.model.vocab_size}.")
                    self.log_train(f"Loaded token mappings from {token_map_path}\n")
                    self.display_token_map()
                except FileNotFoundError:
                    self.log_train("No token mappings file found.\n")
                except ValueError as ve:
                    err_msg = f"Token mapping size mismatch:\n{ve}"
                    self.log_train(err_msg + "\n")
                    self.error_logger.error(err_msg)
                    messagebox.showerror("Token Mapping Error", err_msg)
                    return
                messagebox.showinfo("Model Loaded", f"Model loaded from {load_path}")
        except Exception as e:
            err_msg = f"Load model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            self.error_logger.error(err_msg)
            messagebox.showerror("Load Error", err_msg)

    def run_inference(self):
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
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):
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
            self.error_logger.error(err_msg)
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
                if len(self.token_to_id) != self.model.vocab_size:
                    raise ValueError(f"Loaded token mapping size {len(self.token_to_id)} does not match model's vocab_size {self.model.vocab_size}.")
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            self.error_logger.error(err_msg)
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):
        if not self.stop_flag.is_set():
            if self.process:
                cpu_usage_val = self.process.cpu_percent(interval=None)
                self.cpu_label.config(text=f"CPU: {cpu_usage_val}%")
            else:
                self.cpu_label.config(text="CPU: psutil not installed")
            gpu_usage_val = get_gpu_usage()
            self.gpu_label.config(text=f"GPU: {gpu_usage_val}")
            self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
        if not self.stop_flag.is_set():
            with self.time_lock:
                start_time = self.time_data.get('start_time')
                if start_time is not None:
                    elapsed = time.time() - start_time
                    hours, rem = divmod(elapsed, 3600)
                    minutes, seconds = divmod(rem, 60)
                    if hours >= 1:
                        elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    elif minutes >= 1:
                        elapsed_str = f"{int(minutes)}m {int(seconds)}s"
                    else:
                        elapsed_str = f"{int(seconds)}s"
                    remaining = self.time_data.get('remaining', 0)
                    if remaining > 0:
                        hours_r, rem_r = divmod(remaining, 3600)
                        minutes_r, seconds_r = divmod(rem_r, 60)
                        if hours_r >= 1:
                            remaining_str = f"{int(hours_r)}h {int(minutes_r)}m {int(seconds_r)}s"
                        elif minutes_r >= 1:
                            remaining_str = f"{int(minutes_r)}m {int(seconds_r)}s"
                        else:
                            remaining_str = f"{int(seconds_r)}s"
                    else:
                        remaining_str = "Estimating..."
                else:
                    elapsed_str = "0s"
                    remaining_str = "N/A"
            self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")
            self.master.after(1000, self.update_time_label)

    def evaluate_model(self):
        perplexities = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x], use_residual=True)
            perp = perplexity(logits, y)
            perplexities.append(perp)
        avg_perplexity = np.mean(perplexities)

        hypotheses = []
        references = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x], use_residual=True)
            predicted = np.argmax(logits)
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(y, "<UNK>")])

        bleu_scores_list = []
        for ref, hyp in zip(references, hypotheses):
            bleu_val = bleu_score(ref, hyp)
            bleu_scores_list.append(bleu_val)
        avg_bleu = np.mean(bleu_scores_list)

        self.perplexity_label.config(text=f"Perplexity: {avg_perplexity:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")


def main():
    try:
        root = tk.Tk()
        manager = QuantumChannelManager()
        manager.create_channels(num_channels=100)
        decoder = SubBitDecoder(manager=manager)
        root.qc_manager = manager
        root.decoder = decoder
        gui = QELM_GUI(root)
        multiprocessing.freeze_support()
        root.mainloop()
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.critical(f"Unexpected error:\n{error_trace}")
        hidden_root = tk.Tk()
        hidden_root.withdraw()
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}\n\nCheck the log file for more details.")
        sys.exit(1)


# --------------------------------------------------------------------
# QuantumAttentionLayer and QuantumFeedForwardLayer remain the same
# --------------------------------------------------------------------
class QuantumAttentionLayer(QuantumLayerBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sim_method: str = 'cpu',
        num_threads: int = 1,
        prefix: str = "attn",
        enable_logging: bool = True,
        use_advanced_ansatz: bool = False,
        use_data_reuploading: bool = False
    ):
        super().__init__(sim_method=sim_method, num_threads=num_threads, enable_logging=enable_logging,
                         use_advanced_ansatz=use_advanced_ansatz, use_data_reuploading=use_data_reuploading)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.prefix = prefix
        self.query_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_query")
        self.key_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_key")
        self.value_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_value")
        self.out_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_out")

    def forward(self, x: np.ndarray, mode: str = 'query') -> float:
        if mode == 'query':
            param_store = self.query_params
        elif mode == 'key':
            param_store = self.key_params
        elif mode == 'value':
            param_store = self.value_params
        else:
            param_store = self.out_params
        circuit = self.build_circuit(x, param_store)
        final_state = self.simulate(circuit)
        output = np.abs(final_state[0])**2
        return float(output)

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.query_params.get_values(),
            self.key_params.get_values(),
            self.value_params.get_values(),
            self.out_params.get_values()
        ])

    def set_all_parameters(self, params: np.ndarray):
        total_size = self.query_params.size + self.key_params.size + self.value_params.size + self.out_params.size
        if len(params) != total_size:
            raise ValueError(f"Parameter size mismatch in {self.prefix}. Expected {total_size}, got {len(params)}.")
        offset = 0
        self.query_params.set_values(params[offset:offset + self.query_params.size])
        offset += self.query_params.size
        self.key_params.set_values(params[offset:offset + self.key_params.size])
        offset += self.key_params.size
        self.value_params.set_values(params[offset:offset + self.value_params.size])
        offset += self.value_params.size
        self.out_params.set_values(params[offset:offset + self.out_params.size])

    def to_dict(self) -> dict:
        return {
            "query_params": self.query_params.to_dict(),
            "key_params": self.key_params.to_dict(),
            "value_params": self.value_params.to_dict(),
            "out_params": self.out_params.to_dict()
        }

    def from_dict(self, d: dict):
        self.query_params.from_dict(d["query_params"])
        self.key_params.from_dict(d["key_params"])
        self.value_params.from_dict(d["value_params"])
        self.out_params.from_dict(d["out_params"])


class QuantumFeedForwardLayer(QuantumLayerBase):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        sim_method: str = 'cpu',
        num_threads: int = 1,
        prefix: str = "ffn",
        enable_logging: bool = True,
        use_advanced_ansatz: bool = False,
        use_data_reuploading: bool = False
    ):
        super().__init__(sim_method=sim_method, num_threads=num_threads, enable_logging=enable_logging,
                         use_advanced_ansatz=use_advanced_ansatz, use_data_reuploading=use_data_reuploading)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.prefix = prefix
        self.w1_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_w1")
        self.w2_params = QuantumParameterStore(size=embed_dim, prefix=f"{self.prefix}_w2")

    def forward(self, x: np.ndarray, layer: str = 'w1') -> float:
        if layer == 'w1':
            param_store = self.w1_params
        else:
            param_store = self.w2_params
        circuit = self.build_circuit(x, param_store)
        final_state = self.simulate(circuit)
        output = np.abs(final_state[0])**2
        return float(output)

    def get_all_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.w1_params.get_values(),
            self.w2_params.get_values()
        ])

    def set_all_parameters(self, params: np.ndarray):
        total_size = self.w1_params.size + self.w2_params.size
        if len(params) != total_size:
            raise ValueError(f"Parameter size mismatch in {self.prefix}. Expected {total_size}, got {len(params)}.")
        self.w1_params.set_values(params[:self.w1_params.size])
        self.w2_params.set_values(params[self.w1_params.size:])

    def to_dict(self) -> dict:
        return {
            "w1_params": self.w1_params.to_dict(),
            "w2_params": self.w2_params.to_dict()
        }

    def from_dict(self, d: dict):
        self.w1_params.from_dict(d["w1_params"])
        self.w2_params.from_dict(d["w2_params"])


if __name__ == "__main__":
    main()
