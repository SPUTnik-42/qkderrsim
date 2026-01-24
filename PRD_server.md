# Product Requirements Document (PRD)

## Title

**Server–Client Based BB84 QKD System with Cascade Error Correction**

## Purpose & Scope

The objective of this system is to refactor the existing BB84 simulation into a **true server–client architecture**, where:

* **Alice acts as a server**
* **Bob acts as a client**
* All classical communication occurs via **explicit APIs** (REST / WebSocket / RPC abstraction)
* Quantum transmission, detection, and noise are simulated as modular subsystems
* **Cascade error correction** is implemented strictly through message-based interaction

The system is intended for:

* Experimental evaluation of error correction protocols (Cascade as baseline)
* Visualization and pedagogical demonstration of BB84
* Measuring information leakage, channel usage, and reconciliation efficiency

This is **not** intended to be cryptographically secure, but protocol-faithful.

---

## High-Level Architecture

```
+---------+        Quantum Channel        +---------+        Classical API        +---------+
| Alice   |  --------------------------> | Channel |  -----------------------> | Bob     |
| Server  |                               | / Eve   |                            | Client  |
+---------+                               +---------+                            +---------+
      ^                                                                                |
      |---------------- Classical API (Cascade, Sifting, EC) --------------------------|
```

### Architectural Principles

* Quantum communication is **one-way** (Alice → Bob)
* Classical communication is **bidirectional** and explicit
* No shared memory between Alice and Bob
* All randomness is local (PRNG instances)

---

## Core Components

### 1. Alice (Server)

**Role:**

* BB84 sender
* Authority for parity computation during Cascade

**Responsibilities:**

* PRNG-based bit and basis generation
* Photon preparation
* Quantum transmission initiation
* Classical responses for:

  * Basis reconciliation
  * Parity queries (Cascade)
  * Parameter estimation

**Key Internal Modules:**

* PRNG Engine
* BB84 State Generator
* Classical API Server
* Key Store (raw, sifted)

---

### 2. Bob (Client)

**Role:**

* BB84 receiver
* Cascade driver

**Responsibilities:**

* Measurement basis selection
* Detector event aggregation
* Sifting orchestration
* Cascade protocol execution
* Error correction bookkeeping

**Key Internal Modules:**

* PRNG Engine
* Detector Interface
* Cascade Controller
* Classical API Client
* Metrics Collector

---

### 3. Quantum Channel

**Role:**

* Simulated physical medium

**Responsibilities:**

* Photon loss (attenuation)
* Optical noise (bit flips)
* Optional Eve interception

**Parameters:**

* Fiber length
* Attenuation coefficient
* Optical error probability

---

### 4. Detector System

**Role:**

* Simulated single-photon detector

**Responsibilities:**

* Detection efficiency modeling
* Dark counts
* Basis mismatch randomness

**Outputs:**

* DetectionEvent objects

---

### 5. PRNG System

**Design Requirement:**

* Independent PRNG per actor
* Optional deterministic seeding
* No shared randomness

**Used For:**

* Bit generation
* Basis selection
* Noise modeling
* Eve behavior

---

## Protocol Flow (BB84)

### Phase 1: Quantum Transmission

1. Alice generates (bit, basis)
2. Alice prepares photon
3. Photon sent via quantum channel
4. Bob randomly chooses basis
5. Detector produces detection events

---

### Phase 2: Sifting (Classical)

**API Interaction**

* Bob → Alice: List of detected photon IDs + Bob bases
* Alice → Bob: Matching basis indices

**Result:**

* Sifted keys (Alice & Bob)

---

### Phase 3: Parameter Estimation

**API Interaction**

* Bob selects random subset
* Bob → Alice: Indices to reveal
* Alice → Bob: Corresponding bits

**Metrics Computed:**

* QBER
* Yield

Revealed bits are discarded.

---

## Cascade Error Correction (Server–Client Mapping)

### Design Constraints

* Bob **drives** Cascade
* Alice **only responds** to parity queries
* Each parity request = 1 channel use

---

### Cascade Mapping

| Cascade Step       | Alice (Server)  | Bob (Client)             |
| ------------------ | --------------- | ------------------------ |
| Block partition    | Passive         | Active                   |
| Parity computation | Computes parity | Requests parity          |
| Binary search      | Responds        | Orchestrates             |
| Backtracking       | Responds        | Manages dependency graph |

---

### Required APIs

#### 1. Parity API

```
POST /parity
Request:
{
  "indices": [int, int, ...]
}
Response:
{
  "parity": 0 | 1
}
```

#### 2. Block Parity API

```
POST /block-parity
Request:
{
  "blocks": [[int...], [int...]]
}
Response:
{
  "parities": [0, 1, ...]
}
```

---

### Cascade Metrics (Mandatory)

* Bits revealed
* Errors corrected
* Channel uses (round-trips)

---

## Data Models

### Photon

* id
* bit
* basis
* timestamp

### DetectionEvent

* photon_id
* basis
* measured_bit
* source (signal / dark count)

---

## Non-Functional Requirements

### Modularity

* Each subsystem independently replaceable

### Observability

* Full logging of:

  * Classical messages
  * Error correction steps
  * Information leakage

### Extensibility

* Future EC protocols (LDPC, Winnow)
* Alternative QKD protocols (E91, COW)

---

## Implementation Task Breakdown

### Phase 1 – Refactor Core

* Separate Alice / Bob into isolated processes
* Introduce API boundary
* Remove shared state assumptions

### Phase 2 – Classical Communication Layer

* Define API schema
* Implement client/server wrappers
* Add channel usage accounting

### Phase 3 – BB84 Pipeline

* Quantum send pipeline
* Detector aggregation
* Sifting via API

### Phase 4 – Cascade Integration

* Extract Cascade as protocol driver
* Replace direct parity calls with API calls
* Validate correctness vs current implementation

### Phase 5 – Metrics & Visualization

* Leakage tracking
* Efficiency plots
* Debug tables

---

## Acceptance Criteria

* Alice and Bob run as logically separate entities
* No shared memory access
* Cascade correctness verified
* Channel usage matches theoretical behavior
* System supports Eve injection without code changes

---

## Out of Scope (Explicit)

* Privacy amplification
* Authentication
* Hardware-level modeling
* Cryptographic security guarantees

---

## Deliverables

* Architecture document (this)
* API specification
* Modular implementation
* Experimental scripts
* Visualization outputs

---

**End of Document**
