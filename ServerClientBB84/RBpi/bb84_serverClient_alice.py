import argparse
import asyncio
import datetime
import json
import logging
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from typing import List

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ServerClientBB84.prng import PRNG, Basis
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

def compute_universal_tag(key_bits: List[int], seed: int, tag_bits: int = 64) -> List[int]:
    """
    Computes an Information-Theoretically Secure (ITS) 2-Universal Hash Tag
    using a random binary Toeplitz matrix of dimension (tag_bits x len(key_bits)).
    Collision probability for distinct keys is strictly <= 2^(-tag_bits).
    """
    n = len(key_bits)
    if n == 0:
        return []
    rng = PRNG(seed)
    # A Toeplitz matrix is fully defined by its first column (tag_bits) and first row (n - 1)
    first_col = rng.integers(0, 2, size=tag_bits, dtype=np.uint8)
    first_row = rng.integers(0, 2, size=n - 1, dtype=np.uint8)
    key_arr = np.array(key_bits, dtype=np.uint8)

    # Construct Toeplitz rows via strided slicing and compute dot product over GF(2)
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

class QuantumChannel(Actor):
    def __init__(self, name, length_km, attenuation_db, optical_error_rate, next_actor, seed=None):
        super().__init__(name, seed)
        self.length = length_km
        self.att_db = attenuation_db
        self.optical_error_rate = optical_error_rate
        self.next_actor = next_actor
        self.transmittance = 10**(-(length_km * attenuation_db)/10)

    async def handle_message(self, msg):
        if not isinstance(msg, Photon):
            await self.send(self.next_actor, msg)
            return

        if msg.count == 0:
            return

        p_survive = 1.0 - (1.0 - self.transmittance)**msg.count
        if self.prng.random() > p_survive:
            return

        final_bit = msg.bit
        if self.prng.random() < self.optical_error_rate:
            final_bit = 1 - final_bit

        out_photon = Photon(msg.id, final_bit, msg.basis, msg.creation_time, count=msg.count)
        await self.send(self.next_actor, out_photon)

class ChannelCollector(Actor):
    def __init__(self, name):
        super().__init__(name)
        self.surviving_photons = []
        self.done_event = asyncio.Event()

    async def handle_message(self, msg):
        if isinstance(msg, Photon):
            self.surviving_photons.append((msg.id, msg.bit, msg.basis.value, msg.count))
        elif isinstance(msg, tuple) and msg[0] == "FLUSH":
            self.done_event.set()

class AliceServer(Actor):
    def __init__(self, name, channel: QuantumChannel, collector: ChannelCollector, num_qubits, mu=0.1, verbose=True, seed=None):
        super().__init__(name, seed)
        self.channel = channel
        self.collector = collector
        self.num_qubits = num_qubits
        self.mu = mu
        self.sent_log = {}
        self.sifted_bits = []
        self.final_secret_key = []
        self.key_confirmed = False
        self.verbose = verbose

    async def run_quantum_transmission(self):
        self.sent_log.clear()
        self.collector.surviving_photons.clear()
        self.collector.done_event.clear()
        self.key_confirmed = False
        self.final_secret_key = []

        if self.verbose:
            print(f"\n[ALICE] Generating and sending {self.num_qubits} qubits (mu={self.mu}) through QuantumChannel (L={self.channel.length} km)...")

        sample_log = []
        for i in range(self.num_qubits):
            bit = self.prng.get_bit()
            basis = self.prng.get_basis()
            count = self.prng.poisson(self.mu)
            t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")
            p = Photon(id=i, bit=bit, basis=basis, creation_time=t_now, count=count)
            self.sent_log[i] = p

            if i < 15:
                sample_log.append(f"{bit}({basis.value})")

            await self.send(self.channel, p)

        await self.send(self.channel, ("FLUSH",))
        await self.collector.done_event.wait()

        if self.verbose:
            print(f"[ALICE] First 15 Qubits Sent: {', '.join(sample_log)} ...")
            print(f"[ALICE] Channel exit: {len(self.collector.surviving_photons)}/{self.num_qubits} pulses survived fiber attenuation.")

    async def handle_api_request(self, endpoint, data):
        if endpoint == "/start-quantum":
            await self.run_quantum_transmission()
            return {
                "num_qubits": self.num_qubits,
                "mu": self.mu,
                "transmittance": self.channel.transmittance,
                "photons": self.collector.surviving_photons
            }
        elif endpoint == "/sift":
            return self._api_sift(data)
        elif endpoint == "/sample":
            return self._api_sample(data)
        elif endpoint == "/parity":
            return self._api_parity(data)
        elif endpoint == "/block-parity":
            return self._api_block_parity(data)
        elif endpoint == "/discard-indices":
            return self._api_discard_indices(data)
        elif endpoint == "/key-confirmation":
            return self._api_key_confirmation(data)
        elif endpoint == "/privacy-amplification":
            return self._api_pa(data)
        else:
            raise ValueError(f"Unknown endpoint {endpoint}")

    def _api_sift(self, data):
        bob_bases = data["bases"]
        if self.verbose:
            print(f"[ALICE] Received sifting request with {len(bob_bases)} bases from Bob.")

        matching_indices = []
        self.sifted_bits = []

        for pid, b_basis_str in bob_bases:
            if pid in self.sent_log:
                tx_photon = self.sent_log[pid]
                if tx_photon.basis.value == b_basis_str:
                    matching_indices.append(pid)
                    self.sifted_bits.append(tx_photon.bit)

        if self.verbose:
            print(f"[ALICE] Bases matched: {len(matching_indices)}. Discarded: {len(bob_bases) - len(matching_indices)}.")
            if self.sifted_bits:
                print(f"[ALICE] Sifted Key Preview: {self.sifted_bits[:20]}...")

        return {"matching_indices": matching_indices}

    def _api_sample(self, data):
        indices = data["indices"]
        if self.verbose:
            print(f"[ALICE] Received QBER sample request for {len(indices)} indices.")

        revealed_bits = [
            self.sifted_bits[idx] if 0 <= idx < len(self.sifted_bits) else None
            for idx in indices
        ]

        indices_set = set(indices)
        self.sifted_bits = [b for i, b in enumerate(self.sifted_bits) if i not in indices_set]

        if self.verbose:
            print(f"[ALICE] Revealed bits: {revealed_bits[:15]}...")
            print(f"[ALICE] Removing revealed bits. Working key size: {len(self.sifted_bits)}.")

        return {"bits": revealed_bits}

    def _par_calc(self, indices: List[int]) -> int:
        p_val = 0
        N = len(self.sifted_bits)
        for idx in indices:
            if 0 <= idx < N:
                p_val ^= self.sifted_bits[idx]
        return p_val

    def _api_parity(self, data):
        return {"parity": self._par_calc(data["indices"])}

    def _api_block_parity(self, data):
        return {"parities": [self._par_calc(b) for b in data["blocks"]]}

    def _api_discard_indices(self, data):
        drop_set = set(data.get("indices", []))
        if drop_set:
            self.sifted_bits = [b for i, b in enumerate(self.sifted_bits) if i not in drop_set]
        return {"remaining_length": len(self.sifted_bits)}

    def _api_key_confirmation(self, data):
        """
        Real-world ITS Key Confirmation (pre-Privacy Amplification).
        Computes a 2-Universal Toeplitz tag on the reconciled key.
        If Bob's tag does not match, the protocol aborts before Privacy Amplification.
        """
        confirm_seed = data["confirm_seed"]
        tag_bits = data.get("tag_bits", 64)
        bob_tag = data["bob_tag"]

        alice_tag = compute_universal_tag(self.sifted_bits, confirm_seed, tag_bits)
        self.key_confirmed = (alice_tag == bob_tag) and (len(self.sifted_bits) > 0)

        if self.verbose:
            tag_hex = hex(int("".join(map(str, alice_tag)), 2))[2:].zfill(tag_bits // 4) if alice_tag else "none"
            print(f"\n[ALICE] Universal Key Confirmation ({tag_bits}-bit 2-Universal Toeplitz Tag):")
            print(f"        Alice Tag: {tag_hex} | Confirmed: {self.key_confirmed} (eps <= 2^-{tag_bits})")
            if not self.key_confirmed:
                print("[ALICE] WARNING: Key Confirmation FAILED! Residual errors detected. Aborting key generation.")
                self.sifted_bits = []

        return {"confirmed": self.key_confirmed}

    def _api_pa(self, data):
        if not self.key_confirmed:
            if self.verbose:
                print("[ALICE] Refusing Privacy Amplification: Reconciled key was not confirmed.")
            self.final_secret_key = []
            return {"status": "aborted_unconfirmed", "pa_key_length": 0}

        seed = data["seed"]
        qber = data["qber"]
        pa_protocol = data.get("pa_protocol", "toeplitz")
        q1 = data.get("q1", 1.0)
        ec_leakage = data.get("ec_leakage", 0)

        if self.verbose:
            print(f"[ALICE] Applying Privacy Amplification ({pa_protocol}) with seed {seed}, QBER {qber:.4%}, q1={q1:.4f}, Total Leakage={ec_leakage} bits")

        pa = PrivacyAmplification(self.sifted_bits, qber, q1=q1, ec_leakage=ec_leakage)
        self.final_secret_key = pa.apply_hash(seed, algorithm=pa_protocol)

        if self.verbose and len(self.final_secret_key) > 0:
            pa_preview = "".join(map(str, self.final_secret_key[:50]))
            print(f"[ALICE] Final Shared Secret ({pa_protocol.upper()}) [Len={len(self.final_secret_key)}] first 50 bits: {pa_preview}")

        return {"status": "success", "pa_key_length": len(self.final_secret_key)}

# --- FRAMED STREAM TRANSPORT (WiFi TCP & LoRa UART) ---

async def read_msg(reader: asyncio.StreamReader) -> dict:
    header = await reader.readexactly(4)
    (length,) = struct.unpack("!I", header)
    compressed = await reader.readexactly(length)
    return json.loads(zlib.decompress(compressed).decode('utf-8'))

async def write_msg(writer: asyncio.StreamWriter, payload: dict):
    raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    compressed = zlib.compress(raw)
    writer.write(struct.pack("!I", len(compressed)) + compressed)
    await writer.drain()

async def handle_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, alice: AliceServer):
    print("[ALICE] Bob connected.")
    try:
        while True:
            req = await read_msg(reader)
            endpoint = req.get("endpoint")
            if endpoint == "/close":
                print("[ALICE] Session closed by Bob.")
                break
            resp = await alice.handle_api_request(endpoint, req.get("data", {}))
            await write_msg(writer, resp)
    except asyncio.IncompleteReadError:
        print("[ALICE] Connection closed.")
    finally:
        writer.close()

async def main():
    parser = argparse.ArgumentParser(description="Alice BB84 Node (WiFi / LoRa) - Literature Experimental Defaults")
    parser.add_argument("--mode", choices=["wifi", "lora"], default="wifi")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--qubits", type=int, default=300000, help="Total weak coherent pulses (N)")
    parser.add_argument("--mu", type=float, default=0.1, help="Mean photon flux per pulse (mu = 0.1)")
    parser.add_argument("--length-km", type=float, default=10.0, help="SMF-28 fiber length in km")
    parser.add_argument("--attenuation-db", type=float, default=0.2, help="Fiber attenuation in dB/km at 1550 nm")
    parser.add_argument("--optical-error", type=float, default=0.01, help="Baseline optical misalignment error e_opt (1%)")
    args = parser.parse_args()

    collector = ChannelCollector("Collector")
    channel = QuantumChannel("Fiber", args.length_km, args.attenuation_db, args.optical_error, next_actor=collector, seed=1003)
    alice = AliceServer("AliceServer", channel, collector, num_qubits=args.qubits, mu=args.mu, verbose=True, seed=1001)

    actors = [collector, channel, alice]
    tasks = [asyncio.create_task(a.start()) for a in actors]

    if args.mode == "wifi":
        server = await asyncio.start_server(
            lambda r, w: handle_stream(r, w, alice), args.host, args.port
        )
        print(f"[ALICE] Listening on WiFi/TCP {args.host}:{args.port} (mu={args.mu}, N={args.qubits}, L={args.length_km}km) ...")
        async with server:
            await server.serve_forever()
    else:
        import serial_asyncio
        print(f"[ALICE] Opening LoRa UART {args.serial_port} @ {args.baudrate} baud...")
        reader, writer = await serial_asyncio.open_serial_connection(url=args.serial_port, baudrate=args.baudrate)
        await handle_stream(reader, writer, alice)

    for a in actors:
        await a.send(a, ("STOP",))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())