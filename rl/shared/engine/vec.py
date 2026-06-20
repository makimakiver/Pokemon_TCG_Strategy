"""Subprocess vec-env: N independent battles, one per process.

``cg.sim.Battle.battle_ptr`` is a process-global singleton (cg/sim.py:67), so a
single process can host exactly one battle at a time. Parallel rollouts therefore
require *process* parallelism, not threads. Each worker owns one ``TCGEnv``; the
main process holds the torch policy and drives the workers gym-style.

Observations have variable option counts (pointer net), so we don't stack them —
``reset``/``step`` return a list of per-env obs dicts. Always runs inside the
Docker engine image.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable

from rl.shared.engine.env import TCGEnv, STOP


def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env: TCGEnv = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset(data))           # data = scenario or None
            elif cmd == "step":
                action = STOP if data == "STOP" else data
                obs, reward, done, info = env.step(action)
                remote.send((obs, reward, done, info))
            elif cmd == "result":
                remote.send((env.done, env.result))
            elif cmd == "close":
                env.close()
                remote.send(None)
                break
            else:
                raise RuntimeError(f"unknown cmd {cmd}")
    except EOFError:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnv:
    def __init__(self, env_fns: list[Callable[[], TCGEnv]]):
        self.n = len(env_fns)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.procs = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(wr, r, fn), daemon=True)
            p.start()
            self.procs.append(p)
            wr.close()

    def reset(self, scenarios=None):
        scenarios = scenarios or [None] * self.n
        for remote, sc in zip(self.remotes, scenarios):
            remote.send(("reset", sc))
        return [remote.recv() for remote in self.remotes]

    def step(self, actions):
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", "STOP" if a is STOP else a))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(infos)

    def results(self):
        for remote in self.remotes:
            remote.send(("result", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
