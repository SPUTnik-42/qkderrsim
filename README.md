# QKD Post-Processing Experimentation

This project is dedicated to experimenting with post-processing techniques in Quantum Key Distribution (QKD). It focuses on simulating and analyzing various error correction and privacy amplification protocols used in QKD systems.

## Overview

Currently, the project performs the following:
*   **BB84 Protocol Simulation**: Simulates the Quantum Bit Error Rate (QBER) and key generation process between Alice and Bob.
*   **Cascade Protocol**: Implements the Cascade error correction protocol to reconcile keys.
*   **LDPC Codes**: Experiments with Low-Density Parity-Check (LDPC) codes for error correction (`qc_ldpc`).
*   **Visualization**: Provides Jupyter notebooks and scripts to visualize the efficiency and performance of different protocols.
*   **Web Interface**: A Django-based web application to interact with and visualize the simulations.

## Project Structure

*   `ActorBasedBB84/`: Contains implementations of the simple BB84 protocol and Cascade using an actor-based model.
*   `ServerClientBB84/`: Contains server-client implementations of BB84, refactored Cascade protocol, and LDPC code experiments.
*   `QKDerrorCorrection/`: The Django web application for the project.
*   `Dev/`: Development scripts and notebooks for testing and visualizing different actor-based approaches and comparisons.
*   `Visualization_*.ipynb`: Jupyter notebooks for visualizing simulation results.

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
