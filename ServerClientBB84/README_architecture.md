# Server-Client BB84 + Cascade Architecture

## Overview
This directory contains the refactored BB84 simulation adhering to the "Server-Client" paradigm.

### Key Components

1.  **Alice (Server)** (`AliceServer` in `bb84_server_client.py`)
    *   Acts as the Quantum Transmitter.
    *   Hosted logic for the "Server" side of the protocol.
    *   Exposes a simulated API for Classical Communication.
    *   **State**: Maintains the `sent_log` and `sifted_bits`.

2.  **Bob (Client)** (`BobClient` in `bb84_server_client.py`)
    *   Acts as the Quantum Receiver.
    *   Drives the protocol execution (Sifting -> Parameter Estimation -> Cascade).
    *   Makes request to Alice via `APIClient`.

3.  **API Layer** (`APIClient`)
    *   Abstracts the network boundary.
    *   Methods correspond to REST endpoints.

4.  **Cascade Algorithm** (`CascadeClientProtocol` in `cascade_refactored.py`)
    *   Refactored to run purely on the Client (Bob).
    *   Uses an `oracle` interface to query parity from the Server (Alice).
    *   Does **not** access Alice's key directly.

## API Specification

The following "endpoints" are simulated in the interaction:

### 1. `POST /sift`
*   **Request**: `{"bases": [(photon_id, basis_enum_value), ...]}`
    *   Bob sends the list of photons he detected and the basis he used.
*   **Response**: `{"matching_indices": [photon_id, ...]}`
    *   Alice responds with the IDs where bases matched.

### 2. `POST /sample`
*   **Request**: `{"indices": [idx1, idx2, ...]}`
    *   Bob selects a random subset of the sifted key indices to reveal for QBER estimation.
*   **Response**: `{"bits": [bit1, bit2, ...]}`
    *   Alice returns the bit values for those indices.
    *   **Side Effect**: Both parties discard these bits from the working key.

### 3. `POST /parity`
*   **Request**: `{"indices": [idx1, idx2, ...]}`
    *   Bob asks for the parity of a specific subset of bits (a block).
*   **Response**: `{"parity": 0 | 1}`
    *   Alice computes XOR sum of bits at those indices.

### 4. `POST /block-parity`
*   **Request**: `{"blocks": [[idx...], [idx...], ...]}`
    *   Batch request for multiple blocks (used in initial Cascade pass).
*   **Response**: `{"parities": [p1, p2, ...]}`

## Protocol Flow

1.  **Quantum Transmission** (One-way): Alice -> Channel -> Bob
2.  **Sifting** (API): Bob sends bases -> Alice confirms matches.
3.  **Parameter Estimation** (API): Bob sends sample indices -> Alice reveals bits. Both calculate QBER.
4.  **Error Correction** (Cascade) (API):
    *   Bob partitions key.
    *   Bob requests parities from Alice (API).
    *   Bob performs Binary Search (API requests for subsets).
    *   Bob corrects errors locally.

## Usage

Run the simulation:
```bash
python3 ServerClientBB84/bb84_server_client.py
```
