"""End-to-end proof the rl/ SGS stack trains on the data/Crow corpus.
 Path 1 (D / search_begin): SGS generations over Crow's 5 LOSS replays.
 Path 2 (offline BC):       behavioral cloning over Crow's 5 WIN replays.
"""
import glob
from pathlib import Path
import numpy as np, torch

from rl.policy import PointerPolicy
from rl.sft import collect_offline_traces, train_sft
from rl.targets import replay_to_targets
import rl.outer_loop as ol

print("="*64)
print("PATH 2 — offline behavioral cloning on CROW WINS")
print("="*64)
win_paths = sorted(glob.glob('data/Crow/win_*.json'))
bc = collect_offline_traces(win_paths)          # seat-aware, wins only
print(f"collected {len(bc)} BC (state,pick) pairs from {len(win_paths)} win replays")
policy = PointerPolicy()
train_sft(policy, bc, epochs=2, lr=1e-3)        # prints per-epoch loss

print("\n"+"="*64)
print("PATH 1 — SGS over CROW LOSSES (search_begin scenarios)")
print("="*64)
D = [s for p in sorted(glob.glob('data/loser/lost_crow*.json')) for s in replay_to_targets(p)]
print(f"Crow-only target set |D| = {len(D)}")
ol.build_target_set = lambda *a, **k: D          # isolate Crow losses for the proof
_, hist = ol.run_sgs(generations=2, batch_size=4, run_name="crow_smoke")
print("SGS history:", [(h['gen'], round(h['loss'],3), h['cum_solved']) for h in hist])
print("\nOK — rl/ SGS stack trained on the Crow corpus (both paths).")
