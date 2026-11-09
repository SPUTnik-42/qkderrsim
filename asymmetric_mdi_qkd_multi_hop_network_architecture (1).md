# Software Design & Architecture Document (SDAD)

## Project: Empirical Asymmetric MDI-QKD & Repeaterless Multi-Hop Untrusted-Relay Network Simulation Suite

**Target Modules:** `MDIqkd_server_client.py` and `multihopNet_MDIqkd.py`

**Theoretical Foundations:**

1. *Practical aspects of measurement-device-independent quantum key distribution*, Feihu Xu, Marcos Curty, Bing Qi, and Hoi-Kwong Lo (arXiv:1305.6965 / New J. Phys. 15 113007).

2. *Distance-Security Tradeoffs for Repeaterless End-to-End QKD Networks*, Sumit Chaudhary, Davide Li Calsi, JinHyeock Choi, Marc Geitz, and Janis Nötzel (arXiv:2411.17547).

3. *Existing Codebase Heritage:* `bb84_server_client.py`, `cascade_refactored.py`, `privacy_amplification.py`, `prng.py`.

## 1. Executive Summary & Architectural Philosophy

### 1.1 Core Objective: Empirical Physicality over Analytical Approximation

Most QKD network simulators calculate analytical formulas to output a theoretical Secret Key Rate ($SKR$) number without ever generating a single bit or simulating a single photon detection event. **This software architecture strictly rejects pure analytical shortcuts during runtime execution.**

While analytical equations from Xu et al. (2013) are used in a dedicated **Pre-Transmission Parameter Optimizer Module** to choose optimal laser intensities $\{\mu_a^{opt}, \nu_a^{opt}, \omega_a^{opt}\}$ and $\{\mu_b^{opt}, \nu_b^{opt}, \omega_b^{opt}\}$, the actual QKD execution is **100% empirical and event-driven**:

* Alice and Bob generate discrete weak coherent pulses (WCPs) with randomized phases $\phi \in [0, 2\pi)$, decoy intensity tags, and BB84 polarization states.

* Pulses traverse separate asymmetric optical fibers ($L_{ac} \neq L_{bc}$) subject to unitary polarization drift ($U_1, U_2$) and temporal/spectral mode mismatch ($e_m$).

* The untrusted relay **Charles** physically simulates the $50:50$ Beam Splitter (BS), Polarization Beam Splitters ($PBS_1, PBS_2$ with misalignment $U_3$), and four threshold Single-Photon Detectors ($D_{ch}, D_{cv}, D_{dh}, D_{dv}$) with finite efficiency $\eta_d$ and dark count rate $Y_0$.

* Actual detector clicks are recorded, Bell State Measurements ($\vert{}\psi^-\rangle$ Singlet and $\vert{}\psi^+\rangle$ Triplet) are empirically post-selected, decoy-state gains $Q_{Z,X}^{q_a, q_b}$ and QBERs $E_{Z,X}^{q_a, q_b}$ are counted from raw bit arrays, and actual bitstrings are reconciled via `ParityOracle` protocols (`Cascade`, `Winnow`, `NR_LDPC`, `Polar`) and compressed via `PrivacyAmplification` (`Toeplitz`, `SHAKE256`).

### 1.2 Architectural Decoupling

The software is divided into two primary executable modules and four abstract architectural layers:

1. **`MDIqkd_server_client.py` (Link-Layer Physical & Post-Processing Engine):** Evolves the dual-actor `bb84_server_client.py` into a **Tri-Actor Asymmetric MDI-QKD Engine** (`AliceServer`, `BobClient`, `CharlesRelayActor`). It preserves full backward compatibility with `ParityOracle` so that all existing error correction scripts (`cascade_refactored.py`, `winnow_refactored.py`, `nr_ldpc_standard.py`, `polar_codes.py`) and `privacy_amplification.py` plug in without modification.

2. **`multihopNet_MDIqkd.py` (Network Topology, Routing, Key Management & Optimization Engine):** Implements the repeaterless end-to-end architecture of Chaudhary et al. (2024). It orchestrates point-to-point edge QKD links ($P_{0,1}, P_{m, m+1}$) and long-range $t_j$-hop MDI-QKD sessions ($K_{i, i+d}$ where $2 \leq d \leq t_j$) across $M$ disjoint paths, executes the cascading One-Time Pad (OTP) XOR key-forwarding relay, and provides a multi-objective optimization suite for Distance-Security-Resource tradeoffs.

## 2. System Architecture & Layer Stack

```
+-----------------------------------------------------------------------------------+
| LAYER 4: NETWORK OPTIMIZATION & RESEARCH ANALYTICS (multihopNet_MDIqkd.py)        |
|  - Asymmetric Intensity Optimizer (Xu et al. Flowchart Fig. 6 & Theorem 1)        |
|  - Network Tradeoff Optimizer (Chaudhary et al. P_break^U, SKR_protocol, eta_R)   |
|  - Fault-Injection & Correlated Node Hacking Simulator (Assumptions 1, 2, 3)      |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| LAYER 3: TOPOLOGY, ROUTING & KEY FORWARDING LAYER (multihopNet_MDIqkd.py)         |
|  - Abstract Topology Graph Builder (Line, k-Disjoint, Mesh, Real-World Topologies)|
|  - Abstract Routing Engine (Suurballe / Yen k-Shortest Vertex-Disjoint Paths)     |
|  - ETSI GS QKD 014 Key Management Entity (KME) & Matrix-Based XOR Forwarding      |
|  - Multi-Path Secret Key Distillation: K_final = XOR_{j=1..M} X^(j)               |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| LAYER 2: CLASSICAL POST-PROCESSING & VERIFICATION (MDIqkd_server_client.py)       |
|  - Charles BSM Coincidence Filter & Basis/Intensity Sifting API                   |
|  - 9-Pair Finite Two-Decoy State Parameter Estimator (Y_{Z,L}^{1,1}, e_{X,U}^{1,1})|
|  - Pluggable ParityOracle Reconciliation (Cascade, Winnow, 5G NR-LDPC, Polar)     |
|  - MDI-Adapted Privacy Amplification (Toeplitz Universal_2, SHAKE256)             |
|  - Industry-Standard Key Verification (Wegman-Carter / Poly1305 / SHA3-256 MAC)   |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| LAYER 1: EMPIRICAL QUANTUM OPTICAL PHYSICAL LAYER (MDIqkd_server_client.py)       |
|  - Dual WCP Sources (Alice & Bob) with 3-Intensity Decoy Modulation & Phase Rand. |
|  - Asymmetric Quantum Channels (L_ac, L_bc, x = t_a/t_b) + Unitary Drift (U1, U2) |
|  - Mode Mismatch Engine (Time-Jitter e_t, Spectral, Pulse-Shape -> Total e_m)     |
|  - Untrusted Relay Charles: 50:50 BS + PBS1(U3)/PBS2 + 4 Threshold SPDs           |
+-----------------------------------------------------------------------------------+


```

## 3. Part I: Detailed Design of `MDIqkd_server_client.py`

### 3.1 Evolution from `bb84_server_client.py` to Tri-Actor MDI Architecture

In `bb84_server_client.py`, the topology was unidirectional:

$$
\text{AliceServer} \longrightarrow \text{QuantumChannel} \longrightarrow \text{Detector} \longrightarrow \text{BobClient}
$$

In `MDIqkd_server_client.py`, both Alice and Bob are **Quantum Transmitters (Senders)**, and Charles is an **Untrusted Quantum Measurement Relay** located between them:

$$
\text{AliceServer} \longrightarrow \text{Channel}_{AC} (L_{ac}, t_a, U_1) \longrightarrow \text{CharlesRelayActor} \longleftarrow \text{Channel}_{BC} (L_{bc}, t_b, U_2) \longleftarrow \text{BobClient}
$$

Despite both Alice and Bob being optical transmitters, during the **Classical Post-Processing Phase**, we preserve the exact `AliceServer` (REST-like API Server) and `BobClient` (Orchestrator & `ParityOracle` Consumer) roles from `bb84_server_client.py`. This design choice guarantees that `CascadeClientProtocol`, `WinnowClientProtocol`, `NR_LDPC_Standard_ClientProtocol`, and `PolarClientProtocol` execute identically without knowing whether the underlying raw key came from point-to-point BB84 or a Bell State Measurement at Charles.

### 3.2 Physical Layer Mathematical Model & Empirical Simulation (`practical_MDI_NJP3.tex`)

#### 3.2.1 Weak Coherent Pulse (WCP) State Preparation with General Decoy States

In every clock cycle (pulse ID $i \in \{0, 1, \dots, N_{pulses}-1\}$), Alice and Bob independently prepare phase-randomized weak coherent pulses:

1. **Basis Selection:** Each party selects a basis $\mathcal{B}_a, \mathcal{B}_b \in \{Z \text{ (Rectilinear }, +), X \text{ (Diagonal }, \times)\}$ with probabilities $P_Z$ and $P_X = 1 - P_Z$. Following Lo, Chau, and Ardehali (2005) and Xu et al. Section 2:

   * The $Z$ **basis** ($\{\vert{}H\rangle, \vert{}V\rangle\}$) is used exclusively for **Key Generation**.

   * The $X$ **basis** ($\{\vert{}+\rangle, \vert{}-\rangle\}$ where $\vert{}\pm\rangle = \frac{1}{\sqrt{2}}(\vert{}H\rangle \pm \vert{}V\rangle)$) is used exclusively for **Parameter Testing** (estimating phase error rate $e_X^{1,1}$).

2. **Intensity Selection (3-Intensity Finite Decoy Protocol):**

   * Alice selects an intensity setting $q_a \in \{\mu_a, \nu_a, \omega_a\}$ with probabilities $\{p_{\mu_a}, p_{\nu_a}, p_{\omega_a}\}$, satisfying $\mu_a > \nu_a > \omega_a \geq 0$.

   * Bob selects an intensity setting $q_b \in \{\mu_b, \nu_b, \omega_b\}$ with probabilities $\{p_{\mu_b}, p_{\nu_b}, p_{\omega_b}\}$, satisfying $\mu_b > \nu_b > \omega_b \geq 0$.

   * Crucially, per Xu et al. Section 4, $\omega_a, \omega_b$ are **general weak decoy states** ($\omega \neq 0$, e.g., $\omega = 5 \times 10^{-4}$ due to finite $30\text{ dB}$ extinction ratio of practical intensity modulators), avoiding the unrealistic assumption of a perfect vacuum state.

3. **Random Bit & Phase Randomization:**

   * Alice draws $b_a \in \{0, 1\}$ and a uniform optical phase $\phi_a \sim \mathcal{U}(0, 2\pi)$.

   * Bob draws $b_b \in \{0, 1\}$ and a uniform optical phase $\phi_b \sim \mathcal{U}(0, 2\pi)$.

   * The relative phase arriving at Charles's beam splitter for pulse $i$ is $\phi_i = (\phi_a - \phi_b) \bmod 2\pi$.

```
# Conceptual Data Structure for MDI Optical Pulse Packet
@dataclass
class MDIPulse:
    pulse_id: int
    sender: str               # "ALICE" or "BOB"
    bit: int                  # 0 or 1
    basis: Basis              # Basis.RECTILINEAR (Z) or Basis.DIAGONAL (X)
    intensity_label: str      # "mu" (signal), "nu" (decoy1), "omega" (decoy2)
    mean_photon_num: float    # q_a or q_b
    phase_rad: float          # Uniform [0, 2*pi)
    jones_vector: np.ndarray  # Complex 2D polarization vector [c_H, c_V]^T
    time_mode_overlap: float  # Amplitude alpha = sqrt(1 - e_m)


```

#### 3.2.2 Asymmetric Quantum Channels & Polarization Misalignment Model

Let $L_{ac}$ and $L_{bc}$ be the fiber lengths from Alice to Charles and Bob to Charles, respectively, with attenuation coefficient $\alpha \approx 0.2\text{ dB/km}$. The channel transmittances and the **Asymmetry Parameter** $x$ (Xu et al. Section 6) are:

$$
t_a = 10^{-\frac{\alpha L_{ac}}{10}}, \quad t_b = 10^{-\frac{\alpha L_{bc}}{10}}, \quad x = \frac{t_a}{t_b}
$$

At Charles's station, let $\eta_d$ be the single-photon detector (SPD) quantum efficiency. The effective amplitude parameters arriving at Charles's station for a given pulse pair $(q_a, q_b)$ are:

$$
\gamma_a = \sqrt{q_a t_a \eta_d}, \quad \gamma_b = \sqrt{q_b t_b \eta_d}, \quad \beta = \gamma_a \gamma_b, \quad \gamma = \gamma_a^2 + \gamma_b^2
$$

**Unitary Polarization Misalignment (**$U_1, U_2, U_3$**):**
Per Xu et al. Section 3.1 and Figure 1, `PBS2` is defined as the fundamental reference measurement basis. Polarization rotation in Alice's fiber ($U_1$), Bob's fiber ($U_2$), and Charles's second measurement branch `PBS1` ($U_3$) are modeled by 2D unitary rotation matrices:

$$
U_k = \begin{pmatrix} \cos\theta_k & -\sin\theta_k \\ \sin\theta_k & \cos\theta_k \end{pmatrix}, \quad k \in \{1, 2, 3\}
$$

where each misalignment error is $e_k = \sin^2\theta_k$ and the total systematic polarization misalignment error is:

$$
e_d = \sum_{k=1}^{3} e_k
$$

In the empirical simulation engine, we support two operational modes for $\theta_k$:

1. **Static Misalignment Mode:** Fixed $\theta_k = \arcsin(\sqrt{e_k})$ with default split $e_1 = 0.475 e_d$, $e_2 = 0.475 e_d$, $e_3 = 0.05 e_d$ (matching Footnote 5 of Xu et al.).

2. **Stochastic Gaussian Drift Mode:** For each frame/pulse block, $\theta_k$ is sampled from a Gaussian distribution $\mathcal{N}(0, \sigma_k^2)$ where $\sigma_k = \theta_k^{std} = \arcsin(\sqrt{e_k})$, capturing realistic fiber birefringence fluctuations (Section 3.1 & Appendix B.2.4).

#### 3.2.3 Mode Mismatch Model (Time-Jitter, Spectral, and Pulse Shape)

Per Xu et al. Section 3.2 and Eq. (3), Alice's pulse defines the fundamental spatio-temporal-spectral mode $|T_a\rangle$, while Bob's arriving pulse is decomposed into an overlapping mode $|T_a\rangle$ and an orthogonal non-overlapping mode $|\overline{T_a}\rangle$:

$$
|\phi_a\rangle = |T_a\rangle, \quad |\phi_b\rangle = \sqrt{1 - e_m}|T_a\rangle + \sqrt{e_m}|\overline{T_a}\rangle
$$

where $e_m \in [0, 1]$ is the **total mode mismatch probability** (combining time-jitter $e_t$, spectral mismatch, and pulse-shape mismatch).

* **Physical Consequence at the 50:50 Beam Splitter:** The overlapping component ($\sqrt{1-e_m}$) undergoes two-photon Hong-Ou-Mandel (HOM) quantum interference at Charles's $50:50$ BS. The orthogonal component ($\sqrt{e_m}$) transmits/reflects through the $50:50$ BS **classically without interference**.

* As proved in Xu et al. Section 3.2 (and Footnote 7), in a polarization-encoding MDI-QKD system (when $\eta_d \gg Y_0$), mode mismatch $e_m$ **does not increase the** $Z$**-basis QBER (**$E_Z$**)** because two identical $Z$-basis states ($HH$ or $VV$) only trigger horizontal or vertical detectors regardless of interference. Instead, $e_m$ degrades the HOM dip visibility in the $X$ basis, directly increasing $E_X$ and $e_X^{1,1}$.

#### 3.2.4 Universal Empirical Optical Interference & Threshold Detector Click Engine at Charles

Instead of hardcoding separate formulas for $Z$ and $X$ bases, `CharlesRelayActor` implements a **Universal Coherent Mode-Decomposition Interference Engine** that empirically reproduces Eqs. (B.2)–(B.10) of Xu et al. for **any** incident polarization states, misalignment angles $(\theta_1, \theta_2, \theta_3)$, relative phase $\phi$, beam-splitter reflectivity $R_{BS}$ (default $0.5$, or $0.5007$ for realistic Newport coupler per Appendix A.3), and mode mismatch $e_m$:

1. **Incoming State Vectors:** Let Alice's prepared polarization state be $\vec{\psi}_a = (\psi_{a,H}, \psi_{a,V})^T$ and Bob's be $\vec{\psi}_b = (\psi_{b,H}, \psi_{b,V})^T$, normalized to unit length.

2. **Channel Rotation:** After passing through fibers with unitary misalignments $U_1(\theta_1)$ and $U_2(\theta_2)$ and scaling by $\gamma_a = \sqrt{q_a t_a \eta_d}$ and $\gamma_b = \sqrt{q_b t_b \eta_d}$:

   $$
   \vec{E}_a = \gamma_a U_1(\theta_1) \vec{\psi}_a, \quad \vec{E}_b = \gamma_b e^{i\phi} U_2(\theta_2) \vec{\psi}_b
   $$

3. **Mode Decomposition at the 50:50 Beam Splitter:**
   We decompose Bob's field into the coherent overlapping mode $\vec{E}_b^{\parallel} = \sqrt{1 - e_m}\vec{E}_b$ and the incoherent orthogonal mode $\vec{E}_b^{\perp} = \sqrt{e_m}\vec{E}_b$.
   For a beam splitter with transmittance $T_{BS} = 1 - R_{BS}$ (ideally $T_{BS} = R_{BS} = \frac{1}{2}$), the outgoing spatial arms are $c$ (left branch, heading to `PBS1` with misalignment $U_3(\theta_3)$) and $d$ (right branch, heading to reference `PBS2`):

   * **Arm** $c$ **(via** $U_3$**):**

     $$
     \vec{E}_{c}^{\parallel} = U_3(\theta_3) \left( \sqrt{T_{BS}}\vec{E}_a - \sqrt{R_{BS}}\vec{E}_b^{\parallel} \right), \quad \vec{E}_{c}^{\perp} = U_3(\theta_3) \left( -\sqrt{R_{BS}}\vec{E}_b^{\perp} \right)
     $$

   * **Arm** $d$ **(Reference `PBS2`):**

     $$
     \vec{E}_{d}^{\parallel} = \sqrt{R_{BS}}\vec{E}_a + \sqrt{T_{BS}}\vec{E}_b^{\parallel}, \quad \vec{E}_{d}^{\perp} = \sqrt{T_{BS}}\vec{E}_b^{\perp}
     $$

4. **Exact Mean Photon Numbers at the 4 SPDs (**$D_{ch}, D_{cv}, D_{dh}, D_{dv}$**):**
   Because the orthogonal temporal/spectral mode $\perp$ does not interfere with the fundamental mode $\parallel$, the total optical intensity (mean photon number) impinging on each threshold detector for pulse $i$ is the sum of intensities of the two orthogonal modes:

   $$
   |A|^2 = I(D_{ch}) = |E_{c,H}^{\parallel}|^2 + |E_{c,H}^{\perp}|^2
   $$

   $$
   |B|^2 = I(D_{cv}) = |E_{c,V}^{\parallel}|^2 + |E_{c,V}^{\perp}|^2
   $$

   $$
   |C|^2 = I(D_{dh}) = |E_{d,H}^{\parallel}|^2 + |E_{d,H}^{\perp}|^2
   $$

   $$
   |D|^2 = I(D_{dv}) = |E_{d,V}^{\parallel}|^2 + |E_{d,V}^{\perp}|^2
   $$

   *(Verification against Xu et al. Appendix B.2.1: When* $\vec{\psi}_a = \vec{\psi}_b = (1, 0)^T$ 

   $$
   HH mode
   $$

   , $e_m = 0$*,* $T_{BS}=R_{BS}=1/2$*,* $\theta_3 = 0$*, and* $\sin^2\theta_1 = \sin^2\theta_2 = e_{d1}$*, these four equations reduce identically to Eq. (B.2) of Xu et al.!)*

5. **Empirical Threshold Detector Click Generation:**
   For each of the four detectors $V \in \{A (D_{ch}), B (D_{cv}), C (D_{dh}), D (D_{dv})\}$, the probability of registering a click during pulse slot $i$ given background/dark count probability $Y_0$ is (Xu et al. Eq. (B.3)):

   $$
   P_V = 1 - (1 - Y_0)\exp\left(-|V|^2\right)
   $$

   `CharlesRelayActor` draws four independent uniform random numbers $u_A, u_B, u_C, u_D \sim \mathcal{U}(0, 1)$ via its `PRNG` and sets boolean click flags:

   $$
   \text{Click}_V = \mathbb{I}(u_V < P_V), \quad V \in \{A, B, C, D\}
   $$

6. **Bell State Measurement (BSM) Coincidence Post-Selection:**
   Charles inspects the 4-tuple $(\text{Click}_A, \text{Click}_B, \text{Click}_C, \text{Click}_D)$. A valid BSM announcement occurs **if and only if** exactly two orthogonal polarization detectors click:

   * **Triplet** $\vert{}\psi^+\rangle = \frac{1}{\sqrt{2}}(\vert{}HV\rangle + \vert{}VH\rangle)$ **Announcement:**
     Triggered when $(D_{ch} \land D_{cv} \land \neg D_{dh} \land \neg D_{dv})$ OR $(\neg D_{ch} \land \neg D_{cv} \land D_{dh} \land D_{dv})$.
     *(In probability expectation, this matches* $Q^{\psi^+} = 2 P_A P_B (1 - P_C)(1 - P_D)$*).*

   * **Singlet** $\vert{}\psi^-\rangle = \frac{1}{\sqrt{2}}(\vert{}HV\rangle - \vert{}VH\rangle)$ **Announcement:**
     Triggered when $(D_{ch} \land D_{dv} \land \neg D_{cv} \land \neg D_{dh})$ OR $(D_{cv} \land D_{dh} \land \neg D_{ch} \land \neg D_{dv})$.
     *(In probability expectation, this matches* $Q^{\psi^-} = 2 P_A P_D (1 - P_B)(1 - P_C)$*).*

   * **All other click patterns (0, 1, 3, 4 clicks, or same-polarization clicks** $D_{ch} \land D_{dh}$ **or** $D_{cv} \land D_{dv}$**):** Discarded as unsuccessful BSM events.

### 3.3 Classical Sifting, Bit-Flip Rules & Empirical Decoy-State Estimation

#### 3.3.1 Classical Sifting & Bit-Flip Alignment Table

After quantum transmission concludes, Charles publishes the list of successful BSM events:

$$
\mathcal{E}_{BSM} = \left\{ (\text{pulse\_id } i, \text{bell\_state } \in \{\text{"PSI\_PLUS"}, \text{"PSI\_MINUS"}\}) \right\}
$$

Bob queries Alice's `/mdi-sift` endpoint with the list of `pulse_id`s in $\mathcal{E}_{BSM}$, along with Bob's basis $\mathcal{B}_b(i)$ and intensity tag $q_b(i)$. Alice responds with the subset of `pulse_id`s where $\mathcal{B}_a(i) == \mathcal{B}_b(i)$ and her corresponding intensity tags $q_a(i)$.

To correlate their raw bits, Bob applies the standard MDI-QKD bit-flip rule depending on the shared basis and Charles's announced Bell state (Lo et al. 2012):

| Shared Basis | Charles BSM Outcome | Alice's Action | Bob's Action | Physical Correlation Before Flip | 
| ----- | ----- | ----- | ----- | ----- | 
| $Z$ **Basis (**$+$**)** | $\vert{}\psi^+\rangle$ or $\vert{}\psi^-\rangle$ | Keep bit $b_a$ | **Flip bit:** $b_b \leftarrow 1 - b_b$ | Anti-correlated ($HV$ or $VH$ causes coincidence) | 
| $X$ **Basis (**$\times$**)** | $\vert{}\psi^+\rangle$ (Triplet) | Keep bit $b_a$ | **Keep bit:** $b_b \leftarrow b_b$ | Correlated ($++$ or $--$ projects onto $\Vert{}\psi^+\rangle$) | 
| $X$ **Basis (**$\times$**)** | $\vert{}\psi^-\rangle$ (Singlet) | Keep bit $b_a$ | **Flip bit:** $b_b \leftarrow 1 - b_b$ | Anti-correlated ($+-$ or $-+$ projects onto $\Vert{}\psi^-\rangle$) | 

#### 3.3.2 Empirical Gain ($Q_{Z,X}^{q_a, q_b}$) and QBER ($E_{Z,X}^{q_a, q_b}$) Measurement

Because there are 3 intensities for Alice $\{\mu_a, \nu_a, \omega_a\}$ and 3 for Bob $\{\mu_b, \nu_b, \omega_b\}$, there are $3 \times 3 = 9$ intensity pairs $(q_a, q_b)$.
Let $N_{sent}^{q_a, q_b, \mathcal{B}}$ be the actual number of pulse pairs sent where Alice chose $(q_a, \mathcal{B})$ and Bob chose $(q_b, \mathcal{B})$.
Let $N_{sift}^{q_a, q_b, \mathcal{B}}$ be the actual number of successful BSM clicks recorded in that bucket.

* **Empirical Gain for pair** $(q_a, q_b)$ **in basis** $\mathcal{B} \in \{Z, X\}$**:**

  $$
  Q_{\mathcal{B}}^{q_a, q_b} = \frac{N_{sift}^{q_a, q_b, \mathcal{B}}}{N_{sent}^{q_a, q_b, \mathcal{B}}}
  $$

* **Empirical QBER for pair** $(q_a, q_b)$ **in basis** $\mathcal{B}$**:**

  * For all $X$**-basis** buckets (testing basis) and all **non-signal** $Z$**-basis** decoy buckets ($(q_a, q_b) \neq (\mu_a, \mu_b)$), Alice and Bob publicly compare 100% of the sifted bits to compute the exact empirical error rate:

    $$
    E_{\mathcal{B}}^{q_a, q_b} = \frac{1}{N_{sift}^{q_a, q_b, \mathcal{B}}} \sum_{k=1}^{N_{sift}^{q_a, q_b, \mathcal{B}}} \left( b_{a,k}^{q_a, q_b, \mathcal{B}} \oplus b_{b,k}^{q_a, q_b, \mathcal{B}} \right)
    $$

  * For the **Signal Key-Generation Bucket** $(q_a = \mu_a, q_b = \mu_b, \mathcal{B} = Z)$, Alice and Bob do **not** reveal the whole key. Exactly as in `bb84_server_client.py` (`/sample` endpoint), Bob samples a random fraction $f_{sample}$ (e.g., $10\% - 20\%$) of indices to estimate the working QBER $E_Z^{\mu_a, \mu_b}$, discards those sampled indices via `/discard-indices`, and retains the remaining unrevealed bits as the **Clean Working Key** (`clean_key_bob` and `alice.sifted_bits`) for error reconciliation!

#### 3.3.3 Finite Two-Decoy-State Analytical Bound Estimator (Xu et al. Section 4 & Table 2)

Using the empirically measured gains $Q_Z^{q_a, q_b}$ and $Q_X^{q_a, q_b}$ and error rates $E_X^{q_a, q_b}$ from the 9 intensity buckets, the `DecoyStateEstimator` computes the lower bound on the single-photon yield $Y_{Z,L}^{1,1}$ and the upper bound on the single-photon phase error rate $e_{X,U}^{1,1}$.

**Step 1: Gaussian Elimination of** $Y_Z^{0,m}$ **and** $Y_Z^{n,0}$
Compute the intermediate quantities $Q_Z^{M1}$ and $Q_Z^{M2}$ (Xu et al. Section 4):

$$
Q_Z^{M1} = Q_Z^{\nu_a \nu_b} e^{\nu_a + \nu_b} + Q_Z^{\omega_a \omega_b} e^{\omega_a + \omega_b} - Q_Z^{\nu_a \omega_b} e^{\nu_a + \omega_b} - Q_Z^{\omega_a \nu_b} e^{\omega_a + \nu_b}
$$

$$
Q_Z^{M2} = Q_Z^{\mu_a \mu_b} e^{\mu_a + \mu_b} + Q_Z^{\omega_a \omega_b} e^{\omega_a + \omega_b} - Q_Z^{\mu_a \omega_b} e^{\mu_a + \omega_b} - Q_Z^{\omega_a \mu_b} e^{\omega_a + \mu_b}
$$

**Step 2: Elimination of** $Y_Z^{1,2}$ **or** $Y_Z^{2,1}$ **(Asymmetric Branching)**

* **Case 1:** If $\frac{\mu_a + \omega_a}{\nu_a + \omega_a} \leq \frac{\mu_b + \omega_b}{\nu_b + \omega_b}$ (Eq. 6 of Xu et al.):

  $$
  Y_{Z,L}^{1,1} = \frac{(\mu_a^2 - \omega_a^2)(\mu_b - \omega_b)Q_Z^{M1} - (\nu_a^2 - \omega_a^2)(\nu_b - \omega_b)Q_Z^{M2}}{(\mu_a - \omega_a)(\mu_b - \omega_b)(\nu_a - \omega_a)(\nu_b - \omega_b)(\mu_a - \nu_a)}
  $$

* **Case 2:** If $\frac{\mu_a + \omega_a}{\nu_a + \omega_a} > \frac{\mu_b + \omega_b}{\nu_b + \omega_b}$ (Eq. 7 of Xu et al.):

  $$
  Y_{Z,L}^{1,1} = \frac{(\mu_b^2 - \omega_b^2)(\mu_a - \omega_a)Q_Z^{M1} - (\nu_b^2 - \omega_b^2)(\nu_a - \omega_a)Q_Z^{M2}}{(\mu_a - \omega_a)(\mu_b - \omega_b)(\nu_a - \omega_a)(\nu_b - \omega_b)(\mu_b - \nu_b)}
  $$

**Step 3: Upper Bound on Single-Photon Phase Error Rate** $e_{X,U}^{1,1}$
First, $Y_{X,L}^{1,1}$ is estimated either via the exact same two-decoy formula applied to $Q_X^{q_a, q_b}$ or via the theoretical equivalence $Y_{X,L}^{1,1} = Y_{Z,L}^{1,1}$ (Xu et al. Footnote 11). Then, from Table 2 of Xu et al.:

$$
e_{X,U}^{1,1} = \frac{e^{\nu_a+\nu_b}Q_X^{\nu_a \nu_b}E_X^{\nu_a \nu_b} + e^{\omega_a+\omega_b}Q_X^{\omega_a \omega_b}E_X^{\omega_a \omega_b} - e^{\nu_a+\omega_b}Q_X^{\nu_a \omega_b}E_X^{\nu_a \omega_b} - e^{\omega_a+\nu_b}Q_X^{\omega_a \nu_b}E_X^{\omega_a \nu_b}}{(\nu_a - \omega_a)(\nu_b - \omega_b) Y_{X,L}^{1,1}}
$$

**Step 4: Finite-Key Statistical Fluctuation Option**
When `enable_finite_key_fluctuations=True`, each empirical observable $Q_{\mathcal{B}}^{q_a, q_b}$ is bounded within $[\underline{Q}_{\mathcal{B}}^{q_a, q_b}, \overline{Q}_{\mathcal{B}}^{q_a, q_b}]$ using Hoeffding/Chernoff confidence intervals parameterized by the security failure probability $\epsilon_{sec} = 10^{-10}$ (Curty et al., arXiv:1307.1081, step 4 of Figure 6 in Xu et al.):

$$
\delta(N_{sift}, \epsilon) = \sqrt{\frac{N_{sift}}{2} \ln\left(\frac{1}{\epsilon}\right)}
$$

The worst-case directional bounds ($\pm \delta$) are substituted into $Q_Z^{M1}, Q_Z^{M2}$, and $e_{X,U}^{1,1}$ to guarantee composable finite-key security.

### 3.4 Standardized Server-Client API & Seamless Integration of EC, PA, and Key Verification

To elevate `bb84_server_client.py` to an industry-grade research interface while keeping `cascade_refactored.py` and `privacy_amplification.py` 100% compatible, we formalize the classical communication protocol using structured Data Transfer Objects (DTOs) and add a mandatory **Key Verification (`KeyConfirmation`)** endpoint.

#### 3.4.1 Upgraded API Endpoint Specification on `AliceMDIServer`

| Endpoint | Request Payload | Response Payload | Purpose & Protocol Phase | 
| ----- | ----- | ----- | ----- | 
| `POST /mdi-sift` | `{"bsm_events": [(pulse_id, bell_state, bob_basis, bob_intensity_label), ...]}` | `{"matched_events": [(pulse_id, basis, q_a_label, q_b_label, bell_state), ...]}` | **Phase 1 (Sifting):** Identifies basis matches across all 9 intensity pairs. | 
| `POST /decoy-test-reveal` | `{"reveal_requests": {bucket_key: [pulse_ids]}}` | `{"revealed_bits": {bucket_key: [bits]}}` | **Phase 2a (Decoy &** $X$**-Basis Estimation):** Reveals $X$-basis and non-signal $Z$-basis bits to compute empirical $Q_{Z,X}^{q_a, q_b}$ and $E_X^{q_a, q_b}$. | 
| `POST /sample` | `{"indices": [idx_1, idx_2, ...]}` | `{"bits": [b_1, b_2, ...]}` | **Phase 2b (**$Z$**-Signal QBER Sampling):** Identical to `bb84_server_client.py`. Samples and discards revealed bits from the $(\mu_a, \mu_b, Z)$ working key. | 
| `POST /parity` | `{"indices": [idx_1, ...]}` | \`{"parity": 0 | 1}\` | 
| `POST /block-parity` | `{"blocks": [[idx_1, ...], ...]}` | \`{"parities": \[0 | 1, ...\]}\` | 
| `POST /discard-indices` | `{"indices": [idx_1, ...]}` | `{"remaining_length": int}` | **Phase 3 (Synchronization):** Drops failed LDPC chunks or Winnow parity bits. | 
| `POST /key-confirmation` | `{"nonce": int, "tag_bits": int, "client_mac": str}` | `{"verified": bool, "server_mac": str}` | **Phase 4 (Industry-Standard Key Verification):** Verifies 100% bit identity before Privacy Amplification using a universal hash tag. | 
| `POST /privacy-amplification` | `{"seed": int, "qber": float, "pa_protocol": str, "q1": float, "ec_leakage": int}` | `{"status": "success", "pa_key_length": int}` | **Phase 5 (Privacy Amplification):** Compresses reconciled key into final secret key. | 

#### 3.4.2 Seamless Bridge from MDI Decoy Parameters to `PrivacyAmplification`

Let us examine how your existing `privacy_amplification.py` calculates the number of bits $t$ that Eve knows and the final key length $R$:

```
# Existing formula in privacy_amplification.py:
self.t = math.ceil(self.N * (1.0 - self.q1) + self.N * h_qber + ec_info_leak)
self.R = self.N - self.t - self.S


```

Now compare this with the fundamental MDI-QKD key rate equation (Xu et al. Eq. (1)):

$$
R_{MDI} = N_{sift}^{\mu_a, \mu_b, Z} \left[ q_{1,1}^Z \left(1 - H_2(e_{X,U}^{1,1})\right) - f_e(E_Z^{\mu_a, \mu_b}) H_2(E_Z^{\mu_a, \mu_b}) \right]
$$

where $q_{1,1}^Z$ is the **fraction of sifted signal-state clicks that originated from true single-photon pairs** $(n=1, m=1)$:

$$
q_{1,1}^Z = \frac{P_Z^{1,1}(\mu_a, \mu_b) \cdot Y_{Z,L}^{1,1}}{Q_Z^{\mu_a, \mu_b}}, \quad \text{with } P_Z^{1,1}(\mu_a, \mu_b) = \mu_a \mu_b e^{-(\mu_a + \mu_b)}
$$

In your existing `PrivacyAmplification` class, the amount of key retained before subtracting EC leakage and security parameter $S$ is:

$$
N - N(1 - q_1) - N H_2(\text{qber}) = N \left( q_1 - H_2(\text{qber}) \right)
$$

Whereas in MDI-QKD, the single-photon privacy term is $N \cdot q_{1,1}^Z (1 - H_2(e_{X,U}^{1,1})) = N \left( q_{1,1}^Z - q_{1,1}^Z H_2(e_{X,U}^{1,1}) \right)$, and the empirical EC leakage is `ec_leakage = revealed` (measured directly from Cascade/LDPC/Winnow/Polar)!
Therefore, we achieve **100% mathematical equivalence** with your existing `PrivacyAmplification` class by passing:

1. `qber` $\longleftarrow e_{X,U}^{1,1}$ (the decoy-estimated single-photon $X$-basis phase error rate), and

2. `q1` $\longleftarrow q_{1,1}^{eff} = q_{1,1}^Z + (1 - q_{1,1}^Z) H_2(e_{X,U}^{1,1})$ (or by adding an optional `mdi_mode=True` flag in `PrivacyAmplification` so that $t = \lceil N(1 - q_{1,1}^Z) + N q_{1,1}^Z H_2(e_{X,U}^{1,1}) + \text{ec\_leakage} \rceil$).
   This means both `toeplitz_hash` and `shake256_hash` work out-of-the-box!

#### 3.4.3 Industry-Standard Key Confirmation (Verification Step)

In real-world QKD deployments (ETSI/ITU-T standards), Alice and Bob **never** use "God-Mode" array comparison to check if error correction succeeded. Instead, after Cascade/LDPC finishes, Bob selects a random one-time salt/nonce and computes a $\tau$-bit (e.g., $\tau = 64$ bits) Universal\$*2\$ Polynomial Hash (or SHAKE256-MAC) over `corrected_key`:
\$$\\text{TAG}*{Bob} = \\text{PolyHash}*{nonce}(\\text{corrected_key}*{Bob}) \\in {0,1}^{\\tau}\$$
Bob sends $(nonce, \text{TAG}_{Bob})$ to `/key-confirmation`. Alice computes $\text{TAG}_{Alice}$ on her `sifted_bits`.

* If $\text{TAG}_{Bob} == \text{TAG}_{Alice}$, verification succeeds with failure probability $\epsilon_{cor} \leq 2^{-\tau} \approx 5.4 \times 10^{-20}$. Both parties add $\tau$ bits to `ec_leakage` (since the tag was transmitted over the public channel) and proceed to Privacy Amplification.

* If $\text{TAG}_{Bob} \neq \text{TAG}_{Alice}$, the frame is aborted and discarded, preventing corrupted keys from poisoning the multi-hop XOR relay!

### 3.5 Asymmetric MDI-QKD Intensity Optimization Engine (Xu et al. Sections 5 & 6, Appendix C)

A core requirement of your project is handling **Asymmetric MDI-QKD** ($L_{ac} \neq L_{bc}$, $x = t_a / t_b \neq 1$). As proved in Xu et al. Section 6:

* **Naive Option 1 (**$\mu_a = \mu_b = O(1)$**):** Causes severe intensity mismatch at Charles ($\mu_a t_a \neq \mu_b t_b$), degrading the HOM dip and inflating $E_Z$ and $E_X$.

* **Naive Option 2 ("Symmetric Choice",** $\mu_a t_a = \mu_b t_b$**):** Forces the longer-arm or shorter-arm intensity to become too small when $x \ll 1$, yielding poor decoy bounds for $Y_{Z,L}^{1,1}$ and $e_{X,U}^{1,1}$.

* **Optimal Choice (**$\{\mu_a^{opt}, \nu_a^{opt}, \omega_a^{opt}\}, \{\mu_b^{opt}, \nu_b^{opt}, \omega_b^{opt}\}$**):** Increases the secret key rate by **80% at** $x = 0.1$ ($\Delta L = 50\text{ km}$) and **over 130%–150% at** $x = 0.017$ (Tokyo QKD network).

We implement the **5-Step Optimization Framework** from Figure 6 of Xu et al. in an `AsymmetricMDIOptimizer` class that can run in two modes:

#### Mode A: Fast Short-Distance Analytical Optimizer (Theorem 1, Appendix C.1)

When dark counts are negligible ($Y_0 \to 0$), Xu et al. Theorem 1 proves that the estimated key rate factorizes as:

$$
R_{est} = \frac{t_b^2 \eta_d^2}{2} G(x, \mu_a, \mu_b)
$$

where $G(x, \mu_a, \mu_b)$ depends **only on the channel transmittance ratio** $x = t_a / t_b$:

$$
G(x, \mu_a, \mu_b) = x \mu_a \mu_b e^{-(\mu_a + \mu_b)} \left[ 1 - H_2\left(e_d - \frac{e_d^2}{2}\right) \right] - \frac{2x\mu_a \mu_b + (\mu_b^2 + x^2 \mu_a^2)(2e_d - e_d^2)}{2} f_e H_2(E_Z)
$$

with the 2nd-order approximated $Z$-basis error rate (Eq. C.1):

$$
E_Z(x, \mu_a, \mu_b) = \frac{(\mu_b + x\mu_a)^2 (2e_d - e_d^2)}{2 \left[ 2x\mu_a \mu_b + (\mu_b^2 + x^2 \mu_a^2)(2e_d - e_d^2) \right]}
$$

Maximizing $G(x, \mu_a, \mu_b)$ via `scipy.optimize.minimize` (L-BFGS-B / Nelder-Mead) takes $< 2\text{ ms}$ and yields the exact asymptotic optimal signal intensities $(\mu_a^{opt}, \mu_b^{opt})$.

#### Mode B: Full Rigorous Bessel-Function + Two-Decoy Finite-Key Optimizer (Appendix B & Section 5)

For arbitrary distances where dark counts $Y_0$ and mode mismatch $e_m$ matter, the optimizer evaluates the exact Bessel-function equations (Xu et al. Eqs. B.1, B.4, B.6, B.8):

1. Computes exact analytical $Q_Z^{q_a, q_b}, E_Z^{q_a, q_b}, Q_X^{q_a, q_b}, E_X^{q_a, q_b}$ for all 9 pairs $(q_a, q_b) \in \{\mu_a, \nu_a, \omega_a\} \times \{\mu_b, \nu_b, \omega_b\}$ using the modified Bessel function of the first kind $I_0(\cdot)$:

   $$
   Q_Z^{HH, \psi^+} = 2e^{-\frac{\gamma}{2}}(1-Y_0)^2 \left[ I_0(\beta) + (1-Y_0)^2 e^{-\frac{\gamma}{2}} - (1-Y_0)e^{-\frac{\gamma(1-e_{d1})}{2}}I_0(e_{d1}\beta) - (1-Y_0)e^{-\frac{\gamma e_{d1}}{2}}I_0(\beta - e_{d1}\beta) \right]
   $$

   $$
   Q_Z^{HH, \psi^-} = 2e^{-\frac{\gamma}{2}}(1-Y_0)^2 \left[ I_0(\beta - 2\beta e_{d1}) + (1-Y_0)^2 e^{-\frac{\gamma}{2}} - (1-Y_0)e^{-\frac{\gamma(1-e_{d1})}{2}}I_0(e_{d1}\beta) - (1-Y_0)e^{-\frac{\gamma e_{d1}}{2}}I_0(\beta - e_{d1}\beta) \right]
   $$

   $$
   Q_Z^{HV, \psi^+} = 2e^{-\frac{\gamma}{2}}(1-Y_0)^2 \left[ I_0(2\lambda) + (1-Y_0)^2 e^{-\frac{\gamma}{2}} - (1-Y_0)e^{-\frac{\omega_{aux}}{2}}I_0(\lambda) - (1-Y_0)e^{-\frac{\gamma-\omega_{aux}}{2}}I_0(\lambda) \right]
   $$

   $$
   Q_Z^{HV, \psi^-} = 2e^{-\frac{\gamma}{2}}(1-Y_0)^2 \left[ 1 + (1-Y_0)^2 e^{-\frac{\gamma}{2}} - (1-Y_0)e^{-\frac{\omega_{aux}}{2}}I_0(\lambda) - (1-Y_0)e^{-\frac{\gamma-\omega_{aux}}{2}}I_0(\lambda) \right]
   $$

   where $\lambda = \gamma_a \gamma_b \sqrt{e_{d1}(1-e_{d1})}$ and $\omega_{aux} = \gamma_a^2 + e_{d1}(\gamma_b^2 - \gamma_a^2)$.

2. Passes these 9 gains/QBERs through the Two-Decoy estimator (Table 2 of Xu et al.) to obtain $Y_{Z,L}^{1,1}$ and $e_{X,U}^{1,1}$.

3. Fixes $\omega_a^{opt} = \omega_b^{opt} = \omega_{min} = 5 \times 10^{-4}$ (30 dB modulator extinction limit) and optimizes the 4-dimensional vector $\vec{\theta} = (\mu_a, \nu_a, \mu_b, \nu_b)$ subject to $\mu_a > \nu_a > \omega_a$ and $\mu_b > \nu_b > \omega_b$ to maximize $R_{MDI}$.

4. Feeds the resulting optimal intensities $\{\mu_a^{opt}, \nu_a^{opt}, \omega_a^{opt}\}$ and $\{\mu_b^{opt}, \nu_b^{opt}, \omega_b^{opt}\}$ directly into `AliceMDIServer` and `BobMDIClient` before starting the empirical photon simulation!

## 4. Part II: Detailed Design of `multihopNet_MDIqkd.py`

### 4.1 Theoretical Foundation (`main.tex` — Chaudhary et al., arXiv:2411.17547)

Traditional long-distance QKD networks rely on **Trusted Nodes (TNs)** where each intermediate node $N_i$ establishes a key $K_{i-1, i}$ with its predecessor and $K_{i, i+1}$ with its successor, and forwards the secret key $X$ via $M_i = X \oplus K_{i, i+1}$. **If even a single intermediate node** $N_i$ **is compromised, the adversary immediately learns** $X$**.** Furthermore, end users must house bulky, expensive single-photon detectors (*the Complex Measurement Problem*).

Chaudhary et al. solve both problems simultaneously via **Redundant Key Management with Long-Range MDI/TF-QKD and Disjoint Paths**:

1. **Eliminating End-User Detectors:** All quantum detectors are placed inside untrusted measurement relays (Charles nodes) inside the network core. End users (Alice $N_0$ and Bob $N_{m+1}$) and intermediate key-forwarding nodes $N_1, \dots, N_m$ only need compact laser transmitters.

2. **Feature 1 — Increasing the Collusion Threshold (**$t_j$**) via Longer-Range MDI-QKD Sessions:**
   Consider the $j$-th path between Alice ($N_0^{(j)}$) and Bob ($N_{m_j+1}^{(j)}$) containing $m_j$ intermediate key-forwarding nodes $N_1^{(j)}, \dots, N_{m_j}^{(j)}$.

   * **Edge Point-to-Point Links:** Alice ($N_0^{(j)}$) establishes an edge QKD key $P_{0,1}^{(j)}$ with $N_1^{(j)}$ (implemented in our suite either as a short-range BB84 link via `bb84_server_client` or a 1-hop MDI-QKD link with an edge measurement relay $C_{0,1}$). Similarly, $N_{m_j}^{(j)}$ and Bob ($N_{m_j+1}^{(j)}$) establish $P_{m_j, m_j+1}^{(j)}$.

   * **Long-Range** $t_j$**-Reach MDI-QKD Sessions:** Every pair of nodes $N_i^{(j)}$ and $N_{i'}^{(j)}$ separated by $2 \leq \vert{}i - i'\vert{} \leq t_j$ hops performs an independent MDI-QKD session (using an untrusted measurement relay $C_{i, i'}$ located along the optical path between them) to establish a secret key $K_{i, i'}^{(j)} \in \{0,1\}^n$.

   * Crucially, when $d = \vert{}i - i'\vert{}$ is **odd** (or when inter-node fiber spacings are non-uniform), the physical location of the measurement relay $C_{i, i'}$ sits at an **asymmetric** node station ($L_{ac} \neq L_{bc}$)! As noted in Chaudhary et al. Section "Topological Considerations", for odd $t$, the effective link becomes asymmetric:

     $$
     L_{eff} = 2 \left( \frac{L}{t} \left\lfloor \frac{t}{2} + 1 \right\rfloor \right)
     $$

     Our integration with `MDIqkd_server_client.py` (Xu et al.'s asymmetric MDI engine) handles this physical asymmetry natively and empirically!

3. **Feature 2 — Multiple Disjoint Paths (**$M$**) and Independent Key XORing:**
   Alice and Bob establish $M$ vertex-disjoint paths ($j = 1, \dots, M$). Along each path $j$, Alice generates an independent uniform random bitstring $X^{(j)} \in_R \{0,1\}^n$ and forwards it using the redundant XOR cascade. Upon receiving $X^{(1)}, \dots, X^{(M)}$, Alice and Bob distill the final end-to-end key:

   $$
   K_{final}(A) = K_{final}(B) = \bigoplus_{j=1}^{M} X^{(j)}
   $$

### 4.2 Generalized Redundant Key-Forwarding Protocol & Matrix Representation

Let us trace the exact mathematical execution on the $j$-th path with $m_j$ intermediate nodes and redundancy parameter $t_j \in \{2, \dots, m_j\}$:

#### 4.2.1 Node Key Sets $S_i^{(j)}$

After all link-layer QKD sessions complete and are truncated/padded from their respective Key Management Entity (KME) buffers to target block length $n$ bits:

* **Alice (**$i = 0$**):** Holds key set:

  $$
  S_0^{(j)} = \left\{ P_{0,1}^{(j)} \right\} \cup \left\{ K_{0, d}^{(j)} \;\middle|\; 2 \leq d \leq t_j \right\}
  $$

* **Intermediate Node** $N_i^{(j)}$ **(**$1 \leq i \leq m_j$**):** Holds all keys where $i$ is an endpoint:

  $$
  S_i^{(j)} = \left( \{P_{0,1}^{(j)}\} \text{ if } i=1 \text{ else } \emptyset \right) \cup \left( \{P_{m_j, m_j+1}^{(j)}\} \text{ if } i=m_j \text{ else } \emptyset \right) \cup \left\{ K_{\min(i, i'), \max(i, i')}^{(j)} \;\middle|\; i' \in \{0, \dots, m_j+1\}, \; 2 \leq |i - i'| \leq t_j \right\}
  $$

* **Bob (**$i = m_j + 1$**):** Holds key set:

  $$
  S_{m_j+1}^{(j)} = \left\{ P_{m_j, m_j+1}^{(j)} \right\} \cup \left\{ K_{m_j + 1 - d, m_j + 1}^{(j)} \;\middle|\; 2 \leq d \leq t_j \right\}
  $$

#### 4.2.2 Hop-by-Hop Classical XOR Forwarding Cascade

1. **Initialization at Alice (**$i=0$**):** Alice draws $X^{(j)} \in_R \{0,1\}^n$, defines $M_{-1}^{(j)} := X^{(j)}$, and computes:

   $$
   M_0^{(j)} = X^{(j)} \oplus \bigoplus_{k \in S_0^{(j)}} k
   $$

   Alice sends $M_0^{(j)}$ to $N_1^{(j)}$.

2. **Relay Step at Intermediate Node** $N_i^{(j)}$ **(**$i = 1, \dots, m_j$**):**
   Upon receiving $M_{i-1}^{(j)}$ from $N_{i-1}^{(j)}$, node $N_i^{(j)}$ XORs the message with **all keys in its local key store** $S_i^{(j)}$:

   $$
   M_i^{(j)} = M_{i-1}^{(j)} \oplus \bigoplus_{k \in S_i^{(j)}} k
   $$

   and forwards $M_i^{(j)}$ to $N_{i+1}^{(j)}$.

3. **Final Recovery at Bob (**$i = m_j + 1$**):**
   Upon receiving $M_{m_j}^{(j)}$ from $N_{m_j}^{(j)}$, Bob XORs it with all keys in his local key store $S_{m_j+1}^{(j)}$:

   $$
   K^{(j)}(B) = M_{m_j}^{(j)} \oplus \bigoplus_{k \in S_{m_j+1}^{(j)}} k = X^{(j)} \oplus \bigoplus_{i=0}^{m_j+1} \bigoplus_{k \in S_i^{(j)}} k = X^{(j)}
   $$

   **Why does every intermediate key cancel out at Bob?** Because every bipartite QKD key $K_{u, v}^{(j)}$ (with $0 \leq u < v \leq m_j+1$) belongs to **exactly two** sets: $S_u^{(j)}$ and $S_v^{(j)}$. Therefore, in the telescoping cumulative XOR from $i=0$ to $i=m_j+1$, every key $K_{u, v}^{(j)}$ is XORed twice ($K_{u,v}^{(j)} \oplus K_{u,v}^{(j)} = 0^n$), leaving $K^{(j)}(B) = X^{(j)}$!

#### 4.2.3 Matrix-Based Representation & Link-Failure Resilience Analysis

At any hop $i \in \{0, \dots, m_j\}$, the message $M_i^{(j)}$ traversing the link $N_i^{(j)} \to N_{i+1}^{(j)}$ is the cumulative XOR:

$$
M_i^{(j)} = X^{(j)} \oplus \bigoplus_{u=0}^{i} \bigoplus_{v = i+1}^{\min(u + t_j, m_j + 1)} K_{u, v}^{(j)}
$$

We represent the entire key-forwarding protocol on path $j$ as a binary **Key-Message Incidence Matrix** $\mathbf{A}^{(j)} \in \mathbb{F}_2^{(m_j + 1) \times L_j}$, where:

* Each row $r \in \{0, \dots, m_j\}$ corresponds to the classical message $M_r^{(j)}$.

* Each column $c \in \{1, \dots, L_j\}$ corresponds to one established QKD link key $K_{u,v}^{(j)}$.

* Entry $\mathbf{A}_{r, c}^{(j)} = 1$ if key $K_{u,v}^{(j)}$ is active (un-canceled) inside message $M_r^{(j)}$ (which happens if and only if $u \leq r < v$), and $0$ otherwise.

**Algebraic Collusion & Link-Failure Verifier (`MatrixSecurityAnalyzer`):**
Suppose a subset of intermediate nodes $\mathcal{C} \subset \{1, \dots, m_j\}$ colludes with Eve.

* Eve observes all classical messages $M_0^{(j)}, \dots, M_{m_j}^{(j)}$.

* The colluding nodes $\mathcal{C}$ reveal all keys $K_{u,v}^{(j)}$ where $u \in \mathcal{C}$ or $v \in \mathcal{C}$.

* By removing the columns of $\mathbf{A}^{(j)}$ known to $\mathcal{C}$, obtain the reduced matrix $\mathbf{A}_{reduced}^{(j)}(\mathcal{C})$.

* **Theorem (Matrix Rank Security Criterion):** The colluding coalition $\mathcal{C}$ can recover $X^{(j)}$ **if and only if** the all-ones vector $\mathbf{1}$ is not in the column space of $(\mathbf{A}_{reduced}^{(j)}(\mathcal{C}))^T$, or equivalently, there exists a linear combination of rows of $\mathbf{A}_{reduced}^{(j)}(\mathcal{C})$ over $\mathbb{F}_2$ summing to $\mathbf{0}$ with an odd number of rows!

* Using `numpy`/GF(2) Gaussian elimination on $\mathbf{A}^{(j)}$, our software automatically verifies the exact link-failure properties proven in Chaudhary et al.:

  1. When all links are operational, **at least** $t_j$ **consecutive nodes** must be compromised to break path $j$.

  2. If a single **longest** MDI-QKD link ($\vert{}u - v\vert{} = t_j$) fails, the collusion threshold drops to $t_j - 1$ consecutive nodes.

  3. If a single **edge** link connected to Alice or Bob fails, $t_j - 1$ non-consecutive nodes can break the path.

  4. If **any other intermediate** MDI-QKD link ($2 \leq \vert{}u - v\vert{} < t_j$) fails, the security threshold remains strictly $t_j$ consecutive nodes!

### 4.3 Abstract Topology & Routing Layer Design

To satisfy your requirement for an abstract, optimizable Topology and Routing layer, we design two decoupled sub-layers using the Strategy Design Pattern:

```
+-------------------------------------------------------------------------+
|                    AbstractTopologyLayer (Graph G)                      |
|  Nodes: EndNode (Alice/Bob), ForwardingNode (N_i), UntrustedRelay (C_k) |
|  Edges: OpticalFiberLink(length_km, alpha_db, e_d, e_m, wdm_channels)   |
+-------------------------------------------------------------------------+
           ^                           ^                          ^
           |                           |                          |
+--------------------+     +-----------------------+    +-----------------+
| LineTopology       |     | DisjointParallelTopo  |    | ArbitraryMesh   |
| (M=1, m nodes)     |     | (M paths, m_j nodes)  |    | (Tokyo/NSFNET)  |
+--------------------+     +-----------------------+    +-----------------+

+-------------------------------------------------------------------------+
|                     AbstractRoutingEngine                               |
|  Input: Topology Graph G, Source Alice, Target Bob, Optimization Goal   |
|  Output: RoutePlan = {M Disjoint Paths, {t_j}, Relay Assignments C_uv}  |
+-------------------------------------------------------------------------+
           ^                           ^                          ^
           |                           |                          |
+--------------------+     +-----------------------+    +-----------------+
| SuurballeDisjoint  |     | BottleneckSKRMax      |    | SecurityBound   |
| KShortestRouter    |     | WidestPathRouter      |    | ConstrainedRtr  |
+--------------------+     +-----------------------+    +-----------------+


```

#### 4.3.1 `AbstractTopologyLayer`

Represents the physical fiber infrastructure as a weighted multigraph $G = (\mathcal{V}, \mathcal{E})$:

* **Vertices** $\mathcal{V}$**:** Classified into `NodeRole.END_USER` (Alice, Bob), `NodeRole.KEY_FORWARDER` ($N_i$, equipped with laser transmitters and classical KMEs), and `NodeRole.MEASUREMENT_RELAY` ($C_k$, equipped with 50:50 BS, PBSs, and 4 SPDs). Note that in a physical carrier network, a repeater hut at $80\text{ km}$ spacing can host both a `KEY_FORWARDER` module and a `MEASUREMENT_RELAY` module on separate WDM wavelengths!

* **Built-in Topology Generators:**

  1. `LineTopologyGenerator(m, spacing_km)`: Creates a single chain ($M=1$) of $m$ intermediate nodes with uniform or custom spacing (e.g., $80\text{ km}$).

  2. `DisjointParallelTopologyGenerator(M, m_list, spacing_list)`: Creates $M$ parallel vertex-disjoint chains between Alice and Bob.

  3. `MeshTopologyGenerator(adj_matrix)`: Creates arbitrary metropolitan/backbone optical networks (e.g., Tokyo QKD network, pan-European EuroQCI mesh, or random Waxman/Erdős-Rényi graphs).

#### 4.3.2 `AbstractRoutingEngine` & Path Selection Algorithms

Given a general network graph $G$, source $N_0$ (Alice), and destination $N_{m+1}$ (Bob), the routing layer selects the set of $M$ vertex-disjoint paths $\mathcal{P} = \{P_1, \dots, P_M\}$ and assigns the optimal untrusted relay node $C_{u,v}$ for every $(u, v)$ MDI session ($2 \leq \vert{}u - v\vert{} \leq t_j$):

1. **Algorithm 1: Suurballe's / Bhandari's** $K$**-Shortest Vertex-Disjoint Paths (`MinLossDisjointRouter`):**
   Finds $M$ strictly node-disjoint paths minimizing total optical attenuation $\sum_{j=1}^M \sum_{e \in P_j} \alpha_e L_e$ using node-splitting in a directed residual graph.

2. **Algorithm 2: Max-Min Bottleneck** $SKR$ **Disjoint Router (`MaxBottleneckSKRRouter`):**
   Because the overall protocol secret key rate on path $j$ is limited by the longest $t_j$-hop MDI link:

   $$
   SKR_{\text{path } j} = \min_{0 \leq u < v \leq m_j+1, \; |u-v| \leq t_j} SKR_{MDI}(u, v, C_{u,v}^*)
   $$

   this router assigns edge weights equal to the inverse empirical/analytical MDI key rate of the $t_j$-hop span and uses modified Dijkstra/Yen's algorithm to maximize the bottleneck link capacity.

3. **Algorithm 3: Optimal Measurement Relay Placement (`OptimalRelaySelector`):**
   For any pair of key-forwarding nodes $(N_u^{(j)}, N_v^{(j)})$ separated by $d = v - u$ hops ($2 \leq d \leq t_j$), an intermediate node $N_k^{(j)}$ ($u < k < v$) must act as the untrusted BSM relay Charles.

   * If $d$ is **even**, choosing $k = u + \frac{d}{2}$ yields a symmetric MDI-QKD session ($L_{ac} = L_{bc}$, $x = 1$).

   * If $d$ is **odd** (or link lengths are heterogeneous), $k = u + \lfloor \frac{d}{2} \rfloor$ or $k = u + \lceil \frac{d}{2} \rceil$ produces an **asymmetric MDI-QKD session** ($L_{ac} \neq L_{bc}$, $x = t_a/t_b \neq 1$). The router queries `AsymmetricMDIOptimizer` for each candidate relay $k \in \{u+1, \dots, v-1\}$ and selects the relay $k^*$ that maximizes the asymmetric MDI key rate $R_{MDI}(N_u, N_v; C_k)$!

### 4.4 Security, Resource & Tradeoff Optimization Module (`NetworkTradeoffOptimizer`)

This module implements all analytical and empirical security formulas from Chaudhary et al. (2024) (`main.tex`) to enable automated optimization of $(M, \{m_j\}, \{t_j\}, L)$.

#### 4.4.1 Network Resource & Range Metrics

* **Total Intermediate Nodes:**

  $$
  N^{(\text{nodes})} = \sum_{j=1}^{M} m_j
  $$

* **Total MDI/TF-QKD Links Required (Eq. 6 of Chaudhary et al.):**

  $$
  N^{(\text{TF/MDI})} = \sum_{j=1}^{M} (t_j - 1)\left(m_j - \frac{t_j}{2} + 1\right)
  $$

* **Range Increment Factor** $\eta_R$ **& Total Network Range:**
  Let $L$ be the maximum physical distance of a $t_j$-hop MDI-QKD link. With uniform inter-node spacing $\Delta L = L / t_j$, the range increment factor is:

  $$
  \eta_R = \min_{j \in \{1, \dots, M\}} \frac{m_j + 1}{t_j}, \quad \text{Total Network Range} = \eta_R L
  $$

  Physical realizability requires $2 \leq t_j \leq m_j$, which constrains $\eta_R$:

  $$
  1 + \frac{M}{N^{(\text{nodes})}} \leq \eta_R \leq \frac{N^{(\text{nodes})}/M + 1}{2}
  $$

* **Effective Asymmetric Link Length for Odd** $t_j$**:**

  $$
  L_{\text{eff}}(L, t_j) = \begin{cases} L & \text{if } t_j \text{ is even} \\ 2 \left( \frac{L}{t_j} \left\lfloor \frac{t_j}{2} + 1 \right\rfloor \right) & \text{if } t_j \text{ is odd} \end{cases}
  $$

#### 4.4.2 Adversarial Node-Hacking Security Models (Assumptions 1, 2, and 3)

The `NetworkTradeoffOptimizer` computes the protocol break probability under all three adversarial models defined in Chaudhary et al.:

1. **Model 1: Independent Node Hacking (Assumption 1, Eqs. 1–5):**
   Each intermediate node is independently compromised with probability $p \in [0, 1]$ (e.g., $p = 0.001$).

   * **Exact Combinatorial Break Probability on Path** $j$ **(via Dynamic Programming):**
     Let $C(m_j, t_j, i)$ be the exact number of binary strings of length $m_j$ with Hamming weight $i$ containing at least $t_j$ consecutive $1$s. Then:

     $$
     \mathcal{P}_{\text{break}}^{(j)}(p, m_j, t_j) = \sum_{i=t_j}^{m_j} C(m_j, t_j, i) p^i (1 - p)^{m_j - i}, \quad P_{\text{break}}^{\text{exact}} = \prod_{j=1}^{M} \mathcal{P}_{\text{break}}^{(j)}(p, m_j, t_j)
     $$

   * **Compact Information-Theoretic Upper Bound** $\mathcal{P}_{\text{break}}^{U_j}$ **(Eqs. 3–5):**

     $$
     \mathcal{P}_{\text{break}}^{U_j} = p^{t_j} 2^{m_j H_2(t_j / m_j)}, \quad P_{\text{break}}^U = \prod_{j=1}^{M} \mathcal{P}_{\text{break}}^{U_j} = \prod_{j=1}^{M} p^{t_j} 2^{m_j H_2(t_j / m_j)}
     $$

     where $H_2(x) = -x\log_2(x) - (1-x)\log_2(1-x)$.

2. **Model 2: Cascading Correlated Node Hacking (Assumption 2, Eq. 7):**
   An initial node on path $j$ is compromised with probability $p$, after which each remaining node on the same path is compromised with an elevated conditional probability $\bar{p} > p$:

   $$
   \bar{\mathcal{P}}_{\text{break}}^{U_j} = p \, \bar{p}^{t_j - 1} \, 2^{m_j H_2\left(\frac{t_j - 1}{m_j}\right)} \sum_{u=1}^{m_j - t_j + 1} (1 - p)^{u - 1}, \quad \bar{P}_{\text{break}}^U = \prod_{j=1}^{M} \bar{\mathcal{P}}_{\text{break}}^{U_j}
   $$

3. **Model 3: Sequential Attacker vs. Intrusion Detection System (Assumption 3, Eqs. 8–9):**
   An attacker compromises nodes sequentially with probability $\hat{p}$ per time slot, while a network Intrusion Detection System (IDS) detects each compromised node with probability $q$ per time slot. The probability of compromising $t_j$ consecutive nodes before detection is:

   $$
   \mathbf{P}(t_j, 0) = \prod_{i=1}^{t_j} \frac{\hat{p}(1 - q)^i}{1 - (1 - \hat{p})(1 - q)^i} < \left(\frac{\hat{p}}{q}\right)^{t_j} (1 - q)^{\frac{t_j(1 + t_j)}{2}} \equiv \hat{\mathcal{P}}_{\text{break}}^{U_j}
   $$

#### 4.4.3 Multi-Objective Optimization Formulations

The `NetworkTradeoffOptimizer` exposes three research optimization solvers:

* **Problem A (Hardware-Minimized Design for Target Security & Distance):**

  $$
  \min_{M, \{m_j\}, \{t_j\}} N^{(\text{TF/MDI})} \quad \text{s.t.} \quad P_{\text{break}}^U \leq \epsilon_{\text{target}}, \quad \text{Range} \geq D_{\text{target}}, \quad SKR_{\text{protocol}} \geq R_{\text{min}}
  $$

* **Problem B (Maximum Security Design under Hardware Budget):**

  $$
  \min_{M, \{m_j\}, \{t_j\}} P_{\text{break}}^U \quad \text{s.t.} \quad N^{(\text{nodes})} \leq N_{\max}, \quad N^{(\text{TF/MDI})} \leq L_{\max}, \quad SKR_{\text{protocol}} \geq R_{\text{min}}
  $$

* **Problem C (Pareto Frontier Sweeper):**
  Reproduces Figure 3 and Table 1 of Chaudhary et al. by sweeping $(N^{(\text{nodes})}, \eta_R, M, t)$ and evaluating both the theoretical bounds and the empirical MDI-QKD key rates.

## 5. Class Diagrams & Complete Module Specifications

### 5.1 Class Architecture of `MDIqkd_server_client.py`

```
+--------------------------------------------------------------------------------------------------+
|                                    MDIqkd_server_client.py                                       |
+--------------------------------------------------------------------------------------------------+
| + MDIConfig (dataclass)                                                                          |
|     - L_ac_km: float, L_bc_km: float, alpha_db_km: float                                         |
|     - eta_d: float = 0.145, Y_0: float = 6.02e-6, e_d: float = 0.015, e_m: float = 0.02          |
|     - mu_a, nu_a, omega_a: float | mu_b, nu_b, omega_b: float                                    |
|     - num_pulses: int, ec_protocol: str, pa_protocol: str, auto_optimize_intensities: bool       |
+--------------------------------------------------------------------------------------------------+
| + AsymmetricMDIOptimizer                                                                         |
|     + compute_analytical_gains_qbers(q_a, q_b, config) -> Tuple[Q_Z, E_Z, Q_X, E_X]              |
|     + estimate_two_decoy_bounds(gains_Z, gains_X, qbers_X, intensities) -> Tuple[Y11_Z_L, e11_X] |
|     + optimize_asymptotic_theorem1(x_ratio, e_d, f_e) -> Tuple[mu_a_opt, mu_b_opt]               |
|     + optimize_full_two_decoy(config, N_signals, eps_sec) -> Dict[str, float]                    |
+--------------------------------------------------------------------------------------------------+
| + AsymmetricQuantumChannel(Actor)                                                                |
|     - length_km: float, transmittance: float, theta_std: float, e_m_contribution: float          |
|     + handle_message(pulse: MDIPulse) -> Applies attenuation, unitary U_k rotation, mode tag     |
+--------------------------------------------------------------------------------------------------+
| + CharlesRelayActor(Actor)                                                                       |
|     - bs_reflectivity: float = 0.5, theta_3: float, eta_d: float, Y_0: float                     |
|     - pulse_buffer_alice: Dict[int, MDIPulse], pulse_buffer_bob: Dict[int, MDIPulse]             |
|     - bsm_announcements: List[BSMAnnouncement]                                                   |
|     + _interfere_and_detect(pulse_a: MDIPulse, pulse_b: MDIPulse) -> Optional[BSMAnnouncement]   |
+--------------------------------------------------------------------------------------------------+
| + AliceMDIServer(Actor)                                                                          |
|     - sent_pulses: Dict[int, MDIPulse], sifted_bits: List[int], final_secret_key: List[int]      |
|     + run_quantum_transmission()                                                                 |
|     + handle_api_request(endpoint: str, data: dict) -> dict                                      |
|     - _api_mdi_sift(), _api_decoy_test_reveal(), _api_sample(), _api_parity()                    |
|     - _api_block_parity(), _api_discard_indices(), _api_key_confirmation(), _api_pa()            |
+--------------------------------------------------------------------------------------------------+
| + MDIAPIClient(ParityOracle)                                                                     |
|     + get_parity(indices) -> int                                                                 |
|     + get_parities(blocks) -> List[int]                                                          |
|     + verify_key(nonce, tag_bits, client_mac) -> bool                                            |
|     + apply_privacy_amplification(seed, e11_x, pa_protocol, q11_eff, ec_leakage) -> dict         |
+--------------------------------------------------------------------------------------------------+
| + BobMDIClient(Actor)                                                                            |
|     - ec_protocol: str ("cascade", "winnow", "nr_ldpc_std", "polar")                             |
|     - pa_protocol: str ("toeplitz", "shake256")                                                  |
|     + run_quantum_transmission()                                                                 |
|     + run_classical_post_processing(charles_announcements) -> MDISessionResult                   |
+--------------------------------------------------------------------------------------------------+


```

### 5.2 Class Architecture of `multihopNet_MDIqkd.py`

```
+--------------------------------------------------------------------------------------------------+
|                                    multihopNet_MDIqkd.py                                         |
+--------------------------------------------------------------------------------------------------+
| + NetworkNode                                                                                    |
|     - node_id: str, path_idx: int, hop_idx: int, position_km: float                              |
|     - kme_store: Dict[str, List[int]]  # Maps link_id (e.g. "K_1_3") -> empirical secret key     |
|     + compute_forwarding_message(incoming_msg: List[int]) -> List[int]                           |
+--------------------------------------------------------------------------------------------------+
| + AbstractTopologyLayer (ABC)                                                                    |
|     + build_graph() -> nx.Graph                                                                  |
|     + get_fiber_distance(node_u, node_v) -> float                                                |
|   Subclasses: LineTopology, DisjointPathTopology, ArbitraryMeshTopology                          |
+--------------------------------------------------------------------------------------------------+
| + AbstractRoutingEngine (ABC)                                                                    |
|     + compute_routes(topology, source, target, M, t_list) -> RoutingPlan                         |
|     + select_optimal_charles_relay(path, u_idx, v_idx) -> Tuple[RelayNode, L_ac, L_bc]           |
|   Subclasses: ShortestDisjointRouter, MaxBottleneckSKRRouter, SecurityConstrainedRouter          |
+--------------------------------------------------------------------------------------------------+
| + MatrixSecurityAnalyzer                                                                         |
|     + build_incidence_matrix(m_j: int, t_j: int, failed_links: List[Tuple[int,int]]) -> np.array |
|     + is_compromised_by_coalition(matrix, compromised_nodes: Set[int]) -> bool                   |
|     + find_minimum_collusion_set(matrix) -> Tuple[int, bool, List[Set[int]]]                     |
+--------------------------------------------------------------------------------------------------+
| + NetworkTradeoffOptimizer                                                                       |
|     + calc_exact_break_prob(p, M, m_list, t_list) -> float                                       |
|     + calc_upper_bound_break_prob(p, M, m_list, t_list) -> float                                 |
|     + calc_correlated_break_prob_assump2(p, p_bar, M, m_list, t_list) -> float                   |
|     + calc_sequential_ids_break_prob_assump3(p_hat, q_ids, M, t_list) -> float                   |
|     + optimize_network_parameters(target_range_km, max_p_break, min_skr) -> OptimalNetConfig     |
+--------------------------------------------------------------------------------------------------+
| + MultiHopMDINetworkOrchestrator                                                                 |
|     + establish_all_qkd_links_empirically(routing_plan) -> Dict[str, MDISessionResult]           |
|     + execute_redundant_key_forwarding(key_length_n: int) -> NetworkExecutionReport              |
+--------------------------------------------------------------------------------------------------+


```

## 6. End-to-End Execution Workflows (Step-by-Step)

### 6.1 Workflow 1: Single Asymmetric Link Execution in `MDIqkd_server_client.py`

1. **Initialization & Asymmetric Intensity Optimization:**

   * User specifies $L_{ac} = 50\text{ km}$, $L_{bc} = 10\text{ km}$ ($x = t_a/t_b = 10^{-0.2(50-10)/10} = 10^{-0.8} \approx 0.1585$), $e_d = 0.015$, $e_m = 0.02$, $\eta_d = 0.145$, $Y_0 = 6.02 \times 10^{-6}$.

   * `AsymmetricMDIOptimizer.optimize_full_two_decoy()` computes the optimal asymmetric intensities: e.g., $\mu_a^{opt} \approx 0.41, \nu_a^{opt} \approx 0.07, \omega_a^{opt} = 5 \times 10^{-4}$ and $\mu_b^{opt} \approx 0.07, \nu_b^{opt} \approx 0.01, \omega_b^{opt} = 5 \times 10^{-4}$ (matching Table 3 of Xu et al.).

2. **Simultaneous Empirical Quantum Transmission:**

   * `AliceMDIServer` and `BobMDIClient` concurrently launch `run_quantum_transmission()` for $N_{pulses}$ clock cycles.

   * Each pulse traverses `AsymmetricQuantumChannel_AC` (with $U_1$) and `AsymmetricQuantumChannel_BC` (with $U_2$ and mode mismatch $e_m$) to arrive at `CharlesRelayActor`.

3. **Empirical HOM Interference & Threshold Detection at Charles:**

   * For each pulse pair $(i)$, `CharlesRelayActor` computes the exact 4-detector optical intensities $(|A|^2, |B|^2, |C|^2, |D|^2)$ using the coherent/incoherent mode decomposition and samples threshold detector clicks with efficiency $\eta_d$ and dark count $Y_0$.

   * Whenever a valid $|\psi^+\rangle$ or $|\psi^-\rangle$ coincidence occurs, Charles appends `(pulse_id, bell_state)` to `bsm_announcements`.

4. **Sifting & 9-Bucket Decoy Parameter Estimation:**

   * Bob calls `POST /mdi-sift` on Alice with Charles's announcements.

   * Both parties align their bits using the MDI bit-flip rules (Table in Section 3.3.1) and partition events into the 9 $(q_a, q_b)$ buckets for $Z$ and $X$ bases.

   * Bob calls `POST /decoy-test-reveal` to compute the empirical gains $Q_{Z,X}^{q_a, q_b}$ and $X$-basis error rates $E_X^{q_a, q_b}$, then calculates $Y_{Z,L}^{1,1}$, $e_{X,U}^{1,1}$, and single-photon fraction $q_{1,1}^Z$.

   * Bob calls `POST /sample` on the $(\mu_a, \mu_b, Z)$ bucket to estimate empirical $E_Z^{\mu_a, \mu_b}$ and discards the sampled test bits.

5. **Pluggable Error Correction (`ParityOracle`):**

   * Bob invokes the configured reconciliation protocol (`CascadeClientProtocol`, `WinnowClientProtocol`, `NR_LDPC_Standard_ClientProtocol`, or `PolarClientProtocol`) passing `clean_key_bob`, `est_qber` ($E_Z^{\mu_a, \mu_b}$), and `self.api` (`MDIAPIClient`).

   * The protocol interacts with Alice via `/parity`, `/block-parity`, and `/discard-indices` until Bob obtains `corrected_key` and exact `bits_revealed`.

6. **Key Confirmation & Privacy Amplification:**

   * Bob verifies 100% key agreement via `POST /key-confirmation` (64-bit Universal Hash MAC).

   * Bob initializes `PrivacyAmplification(corrected_key, qber=e11_X_U, q1=q11_eff, ec_leakage=bits_revealed + tag_bits)` and runs `toeplitz_hash` or `shake256_hash`, then synchronizes with Alice via `POST /privacy-amplification`.

### 6.2 Workflow 2: Multi-Hop Disjoint-Path Network Execution in `multihopNet_MDIqkd.py`

1. **Topology & Route Planning:**

   * User instantiates a network (e.g., $M = 2$ disjoint paths, $m_1 = m_2 = 4$ intermediate nodes per path, redundancy parameter $t_1 = t_2 = 3$, inter-node spacing $60\text{ km}$).

   * `AbstractRoutingEngine` enumerates all required QKD sessions per path $j \in \{1, 2\}$:

     * Edge links: $(N_0^{(j)}, N_1^{(j)})$ and $(N_4^{(j)}, N_5^{(j)})$.

     * 2-hop symmetric MDI links ($d=2$, $L_{ac}=60\text{ km}, L_{bc}=60\text{ km}$): $(N_0, N_2)$ via relay $N_1$; $(N_1, N_3)$ via relay $N_2$; $(N_2, N_4)$ via relay $N_3$; $(N_3, N_5)$ via relay $N_4$.

     * 3-hop **asymmetric** MDI links ($d=3$, $L_{ac}=120\text{ km}, L_{bc}=60\text{ km}$, $x = 0.063$): $(N_0, N_3)$ via relay $N_2$; $(N_1, N_4)$ via relay $N_3$; $(N_2, N_5)$ via relay $N_4$.

2. **Empirical Link-Layer Key Generation:**

   * `MultiHopMDINetworkOrchestrator` executes `MDIqkd_server_client` (concurrently via `asyncio.gather`) for every scheduled link.

   * Each link empirically generates a verified, privacy-amplified bitstring $K_{u,v}^{(j)}$ and stores it in the KME buffers of $N_u^{(j)}$ and $N_v^{(j)}$.

   * The orchestrator identifies the bottleneck key length $n = \min_{(u,v,j)} |K_{u,v}^{(j)}|$ (determined by the longest 3-hop asymmetric MDI links) and slices an $n$-bit frame from each KME buffer.

3. **Hop-by-Hop Redundant XOR Key Forwarding (Matching Section "GENERALIZED NETWORK" of Chaudhary et al.):**

   * On each path $j$, Alice ($N_0^{(j)}$) generates a random secret $X^{(j)} \in_R \{0,1\}^n$ and sends:

     $$
     M_0^{(j)} = X^{(j)} \oplus P_{0,1}^{(j)} \oplus K_{0,2}^{(j)} \oplus K_{0,3}^{(j)}
     $$

   * $N_1^{(j)}$ computes and forwards:

     $$
     M_1^{(j)} = M_0^{(j)} \oplus P_{0,1}^{(j)} \oplus K_{1,3}^{(j)} \oplus K_{1,4}^{(j)} = X^{(j)} \oplus K_{0,2}^{(j)} \oplus K_{0,3}^{(j)} \oplus K_{1,3}^{(j)} \oplus K_{1,4}^{(j)}
     $$

   * $N_2^{(j)}$ computes and forwards:

     $$
     M_2^{(j)} = M_1^{(j)} \oplus K_{0,2}^{(j)} \oplus K_{2,4}^{(j)} \oplus K_{2,5}^{(j)} = X^{(j)} \oplus K_{0,3}^{(j)} \oplus K_{1,3}^{(j)} \oplus K_{1,4}^{(j)} \oplus K_{2,4}^{(j)} \oplus K_{2,5}^{(j)}
     $$

   * $N_3^{(j)}$ computes and forwards:

     $$
     M_3^{(j)} = M_2^{(j)} \oplus K_{0,3}^{(j)} \oplus K_{1,3}^{(j)} \oplus K_{3,5}^{(j)} = X^{(j)} \oplus K_{1,4}^{(j)} \oplus K_{2,4}^{(j)} \oplus K_{2,5}^{(j)} \oplus K_{3,5}^{(j)}
     $$

   * $N_4^{(j)}$ computes and forwards to Bob ($N_5^{(j)}$):

     $$
     M_4^{(j)} = M_3^{(j)} \oplus K_{1,4}^{(j)} \oplus K_{2,4}^{(j)} \oplus P_{4,5}^{(j)} = X^{(j)} \oplus K_{2,5}^{(j)} \oplus K_{3,5}^{(j)} \oplus P_{4,5}^{(j)}
     $$

   * Bob ($N_5^{(j)}$) decrypts:

     $$
     K^{(j)}(B) = M_4^{(j)} \oplus K_{2,5}^{(j)} \oplus K_{3,5}^{(j)} \oplus P_{4,5}^{(j)} = X^{(j)}
     $$

4. **Multi-Path Final Key Distillation & Security Verification:**

   * Alice and Bob combine their path keys across all $M$ disjoint paths:

     $$
     K_{final}(A) = \bigoplus_{j=1}^{M} X^{(j)}, \quad K_{final}(B) = \bigoplus_{j=1}^{M} K^{(j)}(B)
     $$

   * The simulator empirically verifies:

     1. **End-to-End Correctness:** $K_{final}(A) == K_{final}(B)$ with $100\%$ bit accuracy.

     2. **Empirical Secrecy & Mutual Information Check:** Computes the empirical Shannon entropy and pairwise correlation between $K_{final}(A)$ and the transcript $\{M_0^{(j)}, \dots, M_{m_j}^{(j)}\}$ combined with any coalition of $< t_j$ consecutive compromised nodes per path, confirming zero information leakage!

## 7. Verification, Validation & Empirical Benchmarking Plan

To ensure research-grade reproducibility, the software suite will include automated verification suites against both papers:

| Benchmark Suite | Reference Paper Target | Empirical / Analytical Verification Metric | 
| ----- | ----- | ----- | 
| **Benchmark 1: Polarization & Mode Mismatch Tolerance** | Xu et al. (2013) Figs. 3 & 5 | Sweep $e_d \in [0, 0.07]$ and $e_m \in [0, 0.80]$. Verify empirical gains $Q_{Z,X}$ and QBERs $E_{Z,X}$ from `CharlesRelayActor` converge to Eqs. (B.4)–(B.8) within $\pm 1/\sqrt{N_{sift}}$. | 
| **Benchmark 2: Asymmetric Intensity Advantage** | Xu et al. (2013) Fig. 8 & Table 3 | Fix $x = 0.1$ ($\Delta L = 50\text{ km}$). Run `MDIqkd_server_client.py` under *Symmetric Choice* ($\mu_a t_a = \mu_b t_b$) vs. *Optimal Choice* ($\mu_a^{opt}, \mu_b^{opt}$). Verify $\sim 80\%$ empirical secret key rate improvement. | 
| **Benchmark 3: Multi-Protocol EC & PA Comparison** | `bb84_server_client.py` Heritage | Benchmark `Cascade`, `Winnow`, `5G NR-LDPC`, and `Polar` codes over empirical MDI sifted keys, logging $f_e = \frac{\text{bits\_revealed}}{N H_2(E_Z)}$, channel uses, execution time, and Toeplitz vs. SHAKE256 throughput. | 
| **Benchmark 4: Multi-Hop Collusion & Link Failure Matrix** | Chaudhary et al. (2024) Section "Generalized Network" | Test all $\binom{m_j}{k}$ node collusion subsets and single-link failures on `multihopNet_MDIqkd.py`. Verify $t_j$ consecutive node requirement and exact degradation rules under longest/edge/interior link drops. | 
| **Benchmark 5: Distance-Security Tradeoff Reproduction** | Chaudhary et al. (2024) Fig. 3 & Table 1 | Reproduce the 5 rows of Table 1 ($N^{(\text{nodes})} \in \{32, 64\}$, $M \in \{1, 2\}$, $t \in \{4, 6\}$, spacing $60\text{–}80\text{ km}$), verifying $N^{(\text{TF/MDI})}$, network range, $SKR_{\text{protocol}}$, and $P_{\text{break}}^U$. | 
