import argparse
import asyncio
import datetime
import json
import logging
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from typing import List

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ServerClientBB84.prng import PRNG, Basis
from ServerClientBB84.cascade_refactored import CascadeClientProtocol, ParityOracle
from ServerClientBB84.nr_ldpc_standard import NR_LDPC_Standard_ClientProtocol
from ServerClientBB84.winnow_refactored import WinnowClientProtocol
from ServerClientBB84.polar_codes import PolarClientProtocol
from ServerClientBB84.privacy_amplification import PrivacyAmplification

logging.basicConfig(level=logging.ERROR, format='%(message)s')

@dataclass
class Photon:
    id: int
    bit: int
    basis: Basis
    creation_time: str
    wavelength_nm: float = 1550.0
    count: int = 1

@dataclass
class DetectionEvent:
    photon_id: int
    basis: Basis
    bit: int
    detection_time: str
    source: str

def compute_universal_tag(key_bits: List[int], seed: int, tag_bits: int = 64) -> List[int]:
    """
    Computes an Information-Theoretically Secure (ITS) 2-Universal Hash Tag
    using a random binary Toeplitz matrix of dimension (tag_bits x len(key_bits)).
    """
    n = len(key_bits)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    first_col = rng.integers(0, 2, size=tag_bits, dtype=np.uint8)
    first_row = rng.integers(0, 2, size=n - 1, dtype=np.uint8)
    key_arr = np.array(key_bits, dtype=np.uint8)

    combined = np.concatenate([first_col[::-1], first_row])
    tag = []
    for i in range(tag_bits):
        row = combined[tag_bits - 1 - i : tag_bits - 1 - i + n]
        tag.append(int(np.dot(row, key_arr) & 1))
    return tag

class Actor:
    def __init__(self, name, seed=None):
        self.name = name
        self.mailbox = asyncio.Queue()
        self._running = False
        self.prng = PRNG(seed)

    async def start(self):
        self._running = True
        while self._running:
            msg = await self.mailbox.get()
            if isinstance(msg, tuple) and msg[0] == "STOP":
                self._running = False
                break
            await self.handle_message(msg)

    async def send(self, recipient: 'Actor', msg):
        await recipient.mailbox.put(msg)

    async def handle_message(self, msg): pass

class Detector(Actor):
    def __init__(self, name, efficiency, dark_count_prob, parent_bob, seed=None):
        super().__init__(name, seed)
        self.eta = efficiency
        self.p_dc = dark_count_prob
        self.parent = parent_bob

    async def handle_message(self, msg):
        if isinstance(msg, tuple) and msg[0] == "FLUSH":
            await self.send(self.parent, ("FLUSH",))
            return

        slot_id, photon = msg
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")
        click = False
        measured_bit = -1
        bob_basis = None
        source = "None"

        if self.prng.random() < self.p_dc:
            click = True
            source = "DarkCount"
            measured_bit = self.prng.get_bit()
            bob_basis = self.prng.get_basis()

        if not click and isinstance(photon, Photon):
            p_detect = 1.0 - (1.0 - self.eta)**photon.count
            if self.prng.random() < p_detect:
                click = True
                source = "Signal"
                bob_basis = self.prng.get_basis()
                if bob_basis == photon.basis:
                    measured_bit = photon.bit
                else:
                    measured_bit = self.prng.get_bit()
                    source = "Signal(BasisMismatch)"

        if click:
            event = DetectionEvent(
                photon_id=slot_id,
                basis=bob_basis,
                bit=measured_bit,
                detection_time=current_time,
                source=source
            )
            await self.send(self.parent, event)

class NetworkAPIClient(ParityOracle):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.tx_bytes = 0
        self.rx_bytes = 0

    async def post(self, endpoint: str, data: dict = None) -> dict:
        payload = {"endpoint": endpoint, "data": data or {}}
        raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        compressed = zlib.compress(raw)
        frame = struct.pack("!I", len(compressed)) + compressed
        self.tx_bytes += len(frame)

        self.writer.write(frame)
        await self.writer.drain()

        if endpoint == "/close":
            return {}

        header = await self.reader.readexactly(4)
        (length,) = struct.unpack("!I", header)
        rx_comp = await self.reader.readexactly(length)
        self.rx_bytes += 4 + length
        return json.loads(zlib.decompress(rx_comp).decode('utf-8'))

    async def get_parity(self, indices: List[int]) -> int:
        resp = await self.post("/parity", {"indices": indices})
        return resp["parity"]

    async def get_parities(self, blocks: List[List[int]]) -> List[int]:
        resp = await self.post("/block-parity", {"blocks": blocks})
        return resp["parities"]

    async def discard_indices(self, indices: List[int]):
        return await self.post("/discard-indices", {"indices": indices})

    async def confirm_key(self, confirm_seed: int, bob_tag: List[int], tag_bits: int = 64) -> bool:
        resp = await self.post("/key-confirmation", {
            "confirm_seed": confirm_seed,
            "bob_tag": bob_tag,
            "tag_bits": tag_bits
        })
        return resp["confirmed"]

    async def apply_privacy_amplification(self, seed: int, qber: float, pa_protocol: str = "toeplitz", q1: float = 1.0, ec_leakage: int = 0):
        return await self.post("/privacy-amplification", {
            "seed": seed, "qber": qber, "pa_protocol": pa_protocol,
            "q1": q1, "ec_leakage": ec_leakage
        })

class BobClient(Actor):
    def __init__(self, name, api_client: NetworkAPIClient, protocol="polar", pa_protocol="toeplitz", confirm_bits=64, seed=None, verbose=True, protocol_params=None):
        super().__init__(name, seed)
        self.api = api_client
        self.protocol = protocol.lower()
        self.pa_protocol = pa_protocol.lower()
        self.confirm_bits = confirm_bits
        self.verbose = verbose
        self.received_events = []
        self.detection_done = asyncio.Event()
        self.protocol_params = protocol_params if protocol_params else {}

    async def handle_message(self, msg):
        if isinstance(msg, DetectionEvent):
            self.received_events.append(msg)
        elif isinstance(msg, tuple) and msg[0] == "FLUSH":
            self.detection_done.set()

    async def run_classical_post_processing(self, num_input_qubits, q1=1.0):
        bases_payload = []
        print(f"\n[BOB] Processing {len(self.received_events)} detection events.")

        valid_evs = [e for e in self.received_events if e.bit != -1]
        sample_valid = [f"{e.bit}({e.basis.value})" for e in valid_evs[:15]]
        print(f"[BOB] First 15 Valid Detections: {', '.join(sample_valid)} ...")

        for ev in self.received_events:
            if ev.bit != -1:
                bases_payload.append((ev.photon_id, ev.basis.value))

        print(f"[BOB] Sending {len(bases_payload)} bases to Alice for sifting...")
        response = await self.api.post("/sift", {"bases": bases_payload})
        matching_ids = set(response["matching_indices"])
        print(f"[BOB] Sifting complete. Bases matched on {len(matching_ids)} events.")

        sifted_key_bob = [ev.bit for ev in self.received_events if ev.photon_id in matching_ids]
        print(f"[BOB] Sifted Key Preview: {sifted_key_bob[:20]}...")

        n_sifted = len(sifted_key_bob)
        sample_size = 0.2
        k = int(n_sifted * sample_size)

        if n_sifted < 10:
            print(f"[BOB] Error: Sifted key too short ({n_sifted}) for QBER estimation.")
            return {"sifted_length": n_sifted, "qber": 1.0, "final_length": 0, "pa_length": 0, "revealed": 0, "corrected": 0, "channel_uses": 0, "key_confirmed": False, "corrected_key": [], "final_secret_key": [], "exec_time": 0}

        if k < 10:
            k = 10

        print(f"[BOB] Sampling {k} bits for QBER estimation...")
        indices_to_sample = self.prng.sample(range(n_sifted), k)
        resp = await self.api.post("/sample", {"indices": indices_to_sample})
        alice_bits = resp["bits"]

        errors = sum(1 for i, idx in enumerate(indices_to_sample) if sifted_key_bob[idx] != alice_bits[i])
        qber = errors / k
        print(f"[BOB] QBER Analysis: {errors} errors in {k} samples. Estimated QBER = {qber:.2%}")

        indices_set = set(indices_to_sample)
        clean_key_bob = [int(bool(b)) for i, b in enumerate(sifted_key_bob) if i not in indices_set]
        print(f"[BOB] Discarding revealed bits. Working Key Size: {len(clean_key_bob)}")

        print(f"[BOB] Initializing {self.protocol.title()} Protocol...")
        t_start = time.time()
        est_qber = qber if qber > 0 else 0.01

        if self.protocol == "cascade":
            passes = int(self.protocol_params.get('num_passes', 4))
            protocol = CascadeClientProtocol(num_passes=passes, verbose=self.verbose)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        elif self.protocol in {"ldpc_5g_std", "nr_ldpc_std", "nr_ldpc_standard"}:
            proto_seed = self.prng.randint(0, 2**32 - 1)
            ldpc_params = {k: self.protocol_params[k] for k in ('rate', 'num_iter', 'max_k', 'target_fer', 'sweep_factor', 'max_sweep_iter', 'hash_bits') if k in self.protocol_params}
            protocol = NR_LDPC_Standard_ClientProtocol(verbose=self.verbose, seed=proto_seed, **ldpc_params)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
            stats = getattr(protocol, "stats", {})
            failed_chunks = stats.get("failed_chunk_indices", [])
            if failed_chunks and getattr(protocol, "_perm", None) is not None:
                bounds = stats.get("chunk_boundaries", [])
                perm = protocol._perm
                drop_orig = []
                for c_idx in failed_chunks:
                    lo, hi = bounds[c_idx], bounds[c_idx + 1]
                    drop_orig.extend(perm[lo:hi].tolist())
                drop_set = set(drop_orig)
                corrected_key = [b for i, b in enumerate(corrected_key) if i not in drop_set]
                await self.api.discard_indices(drop_orig)
        elif self.protocol == "winnow":
            num_passes = int(self.protocol_params.get('num_passes', 4))
            protocol = WinnowClientProtocol(num_passes=num_passes, verbose=self.verbose)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        elif self.protocol == "polar":
            u_fer_target = float(self.protocol_params.get('u_fer_target', 0.01))
            c = float(self.protocol_params.get('c', 0.5))
            protocol = PolarClientProtocol(verbose=self.verbose, u_fer_target=u_fer_target, c=c)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)
        else:
            protocol = CascadeClientProtocol(num_passes=4, verbose=self.verbose)
            corrected_key, revealed, errors_cor, uses = await protocol.run(clean_key_bob, est_qber, self.api)

        # --- Step 4: ITS Key Confirmation (Before Privacy Amplification) ---
        key_confirmed = False
        final_secret_key = []
        pa_length = 0
        pa_time = 0.0
        total_leakage = revealed

        if len(corrected_key) > 0 and qber < 1.0:
            confirm_seed = self.prng.randint(0, 2**32 - 1)
            bob_tag = compute_universal_tag(corrected_key, confirm_seed, self.confirm_bits)
            uses += 1
            # Revealing the confirmation tag over the public channel leaks `confirm_bits` of entropy
            total_leakage = revealed + self.confirm_bits

            key_confirmed = await self.api.confirm_key(confirm_seed, bob_tag, self.confirm_bits)
            tag_hex = hex(int("".join(map(str, bob_tag)), 2))[2:].zfill(self.confirm_bits // 4)
            print(f"\n[BOB] Universal Key Confirmation ({self.confirm_bits}-bit 2-Universal Toeplitz Tag):")
            print(f"      Bob Tag  : {tag_hex} | Confirmed by Alice: {key_confirmed} (Collision Bound <= 2^-{self.confirm_bits})")

            if not key_confirmed:
                print("[BOB] ERROR: Key Confirmation failed! Residual bit errors remain. Aborting Privacy Amplification.")
                corrected_key = []
            else:
                # --- Step 5: Privacy Amplification (accounts for EC leakage + confirmation tag leakage) ---
                print(f"\n[BOB] Starting Privacy Amplification using {self.pa_protocol} (q1={q1:.4f}, Total Leakage={total_leakage} bits)...")
                shared_seed = self.prng.randint(0, 2**32 - 1)
                pa_time_start = time.time()
                pa = PrivacyAmplification(corrected_key, qber, q1=q1, ec_leakage=total_leakage)
                pa.log_metrics()
                final_secret_key = pa.apply_hash(seed=shared_seed, algorithm=self.pa_protocol)
                pa_time = time.time() - pa_time_start
                pa_length = pa.R

                await self.api.apply_privacy_amplification(
                    shared_seed, qber, self.pa_protocol, q1=q1, ec_leakage=total_leakage
                )
                print(f"[BOB] Sent PA seed to Alice. Final ITS Secret Key Length: {pa_length} bits.")

        exec_time = time.time() - t_start
        return {
            "sifted_length": n_sifted,
            "qber": qber,
            "final_length": len(corrected_key),
            "pa_length": pa_length,
            "pa_time": pa_time,
            "revealed": total_leakage,
            "corrected": errors_cor,
            "channel_uses": uses,
            "key_confirmed": key_confirmed,
            "corrected_key": corrected_key,
            "final_secret_key": final_secret_key,
            "exec_time": exec_time
        }

async def main():
    parser = argparse.ArgumentParser(description="Bob BB84 Node (WiFi / LoRa) - Literature Experimental Defaults")
    parser.add_argument("--mode", choices=["wifi", "lora"], default="wifi")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--serial-port", default="/dev/ttyUSB1")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--protocol", default="polar", choices=["cascade", "polar", "winnow", "nr_ldpc_std"])
    parser.add_argument("--pa-protocol", default="toeplitz")
    parser.add_argument("--confirm-bits", type=int, default=64, help="2-Universal hash tag length in bits (default: 64)")
    parser.add_argument("--eta", type=float, default=0.90, help="Detector quantum efficiency (default: 0.90)")
    parser.add_argument("--p-dc", type=float, default=1e-5, help="Dark count probability per gate (default: 1e-5)")
    args = parser.parse_args()

    if args.mode == "wifi":
        print(f"[BOB] Connecting to Alice over WiFi/TCP at {args.host}:{args.port} ...")
        reader, writer = await asyncio.open_connection(args.host, args.port)
    else:
        import serial_asyncio
        print(f"[BOB] Opening LoRa UART {args.serial_port} @ {args.baudrate} baud...")
        reader, writer = await serial_asyncio.open_serial_connection(url=args.serial_port, baudrate=args.baudrate)

    api = NetworkAPIClient(reader, writer)
    bob = BobClient(
        "BobClient", api, protocol=args.protocol, pa_protocol=args.pa_protocol,
        confirm_bits=args.confirm_bits, seed=1002, verbose=True
    )
    detector = Detector("Detector", efficiency=args.eta, dark_count_prob=args.p_dc, parent_bob=bob, seed=1004)

    actors = [detector, bob]
    tasks = [asyncio.create_task(a.start()) for a in actors]

    # 1. Request Quantum Transmission from Alice
    print("[BOB] Requesting quantum pulse train from Alice...")
    q_resp = await api.post("/start-quantum")
    num_qubits = q_resp["num_qubits"]
    mu = q_resp["mu"]
    T = q_resp["transmittance"]
    surviving_map = {
        pid: Photon(id=pid, bit=bit, basis=Basis(basis_str), creation_time="", count=cnt)
        for pid, bit, basis_str, cnt in q_resp["photons"]
    }
    print(f"[BOB] Received {len(surviving_map)} surviving pulses across {num_qubits} clock slots. Running Detector (eta={args.eta}, p_dc={args.p_dc})...")

    for slot_id in range(num_qubits):
        await detector.send(detector, (slot_id, surviving_map.get(slot_id, None)))
    await detector.send(detector, ("FLUSH",))
    await bob.detection_done.wait()

    # 2. Compute single-photon bound q1 (GLLP / Luetkenhaus WCP bound) and run Classical Post-Processing
    p_sig = 1.0 - np.exp(-mu * T * detector.eta)
    p_dark = 2 * detector.p_dc
    p_click = p_sig + p_dark - (p_sig * p_dark)
    p_multi = 1.0 - (1.0 + mu) * np.exp(-mu)
    q1 = max(0.0, 1.0 - (p_multi / p_click)) if p_click > 0 else 0.0

    results = await bob.run_classical_post_processing(num_qubits, q1=q1)

    await api.post("/close")
    writer.close()

    print("\n--- RESULTS ---")
    print(f"Sifted Key Length : {results['sifted_length']}")
    print(f"QBER (Est)        : {results['qber']:.4%}")
    print(f"Single-Photon q1  : {q1:.4f}")
    print(f"Reconciled Length : {results['final_length']}")
    print(f"ITS Key Confirmed : {results['key_confirmed']} (2-Universal Toeplitz, eps <= 2^-{args.confirm_bits})")
    print(f"Total Leakage     : {results['revealed']} bits (EC Parities + {args.confirm_bits}-bit Confirmation Tag)")
    print(f"PA Secret Length  : {results['pa_length']} bits")
    print(f"Corrected Errors  : {results['corrected']}")
    print(f"Channel Uses      : {results['channel_uses']}")
    print(f"Network Traffic   : TX={api.tx_bytes / 1024:.2f} KB | RX={api.rx_bytes / 1024:.2f} KB")
    if results["pa_length"] > 0:
        bob_pa_str = "".join(map(str, results["final_secret_key"][:50]))
        print(f"Final Shared Secret ({bob.pa_protocol.upper()}) first 50 bits: {bob_pa_str}")

    for a in actors:
        await a.send(a, ("STOP",))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())