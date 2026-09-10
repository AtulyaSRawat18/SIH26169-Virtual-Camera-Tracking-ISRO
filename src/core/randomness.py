"""Stable isolated RNG streams derived from experiment identity."""
import hashlib
import numpy as np

SUBSYSTEMS = ("motion", "navigation", "attitude", "vibration", "environment",
              "beacon", "camera_noise", "dropout", "distractor", "manoeuvre")


def derive_seed(master_seed: int, run_index: int, subsystem: str) -> int:
    payload = f"SIH26169|rng-v1|{master_seed}|{run_index}|{subsystem}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class RandomStreams:
    def __init__(self, master_seed: int, run_index: int = 0):
        self.master_seed, self.run_index = int(master_seed), int(run_index)
        self.seeds = {name: derive_seed(master_seed, run_index, name) for name in SUBSYSTEMS}
        self._streams = {name: np.random.default_rng(seed) for name, seed in self.seeds.items()}

    def __getitem__(self, name):
        return self._streams[name]
