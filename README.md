# QKD Post-Processing Experimentation

This project is dedicated to experimenting with post-processing techniques in Quantum Key Distribution (QKD). It focuses on simulating and analyzing various error correction and privacy amplification protocols used in QKD systems.

## Overview

Currently, the project performs the following:
*   **BB84 Protocol Simulation**: Simulates the Quantum Bit Error Rate (QBER) and key generation process between Alice and Bob, including finite-key and asymptotic analyses (e.g., `bb84_finite.py`, `simBB84.py`).
*   **Error Correction (EC) Protocols**: Implements multiple reconciliation protocols including Cascade, Winnow, Low-Density Parity-Check (LDPC) codes (Standard and Rate-Adaptive), and Polar Codes.
*   **Privacy Amplification (PA)**: Implements privacy amplification techniques to reduce Eve's information and extract secure cryptographic keys.
*   **Visualization & Analysis**: Provides extensive Jupyter notebooks to optimize, compare, and visualize the efficiency and Secret Key Rates (SKR) of different protocols over varying distances and photon per pulse ($\mu$) values.
*   **Web Interface**: A Django-based web application to interact with and visualize the simulations.

## Project Structure

*   `PostProcessingToolbox/`: A comprehensive toolbox containing modules for QKD post-processing, including error correction (Cascade, Winnow, LDPC, Polar Codes) and privacy amplification.
*   `ActorBasedBB84/`: Contains implementations of the simple BB84 protocol and Cascade using an actor-based model.
*   `ServerClientBB84/`: Contains server-client implementations of BB84 alongside refactored EC and PA protocols.
*   `QKDerrorCorrection/`: The Django web application for configuring and visualizing QKD error correction simulations.
*   `Dev/`: Development scripts and notebooks for visualizing results and testing alternate approaches.
*   `*.ipynb` (Root Directory Jupyter Notebooks): Comprehensive notebooks for varying use-cases:
    *   **Demonstrations**: `BB84_API_Demo.ipynb`, `Cascade_Demo.ipynb`, `PostProcessingToolbox_Demo.ipynb`.
    *   **Comparisons & PA/EC Analysis**: `Visualization_Comparison_EC.ipynb`, `Visualization_Comparison_PA.ipynb`.
    *   **Simulations & Optimization**: `SKR_Optimization_GridSearch.ipynb`, `finite_vs_asymptotic_qkd.ipynb`, `Finite_vs_ServerClient_Multiple_Protocols.ipynb`, and `Finite_vs_ServerClient_SKR_vs_Distance_Mu.ipynb`.


## Setup and Installation

### 1. Virtual Environment

It is recommended to run this project within a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Dependencies

Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: If you are running the Django application, ensure that `django` and `python-dotenv` are installed, as they might not be explicitly listed in the requirements file.*

```bash
pip install django python-dotenv
```

## Django Web Application

The `QKDerrorCorrection` directory contains a Django project designed to provide a web interface for the QKD error correction simulations.

### Environment Configuration (.env)

The Django project uses a `.env` file to manage configuration settings securely. You need to create a `.env` file in the `QKDerrorCorrection` directory (same level as `manage.py` and `db.sqlite3`) or inside `QKDerrorCorrection/QKDerrCorr/`.

Example `.env` content:

```env
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=127.0.0.1 localhost
```

### Running the Server

To run the Django development server:

1.  Navigate to the Django project directory:
    ```bash
    cd QKDerrorCorrection
    ```

2.  Apply migrations (if setting up for the first time):
    ```bash
    python manage.py migrate
    ```

3.  Run the server:
    ```bash
    python manage.py runserver
    ```

Open your browser and navigate to `http://127.0.0.1:8000/` to access the application.
