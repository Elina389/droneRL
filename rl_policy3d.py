"""
Inference wrappers for the trained 3D PPO policy.

PPOPolicy3D  -- pure learned policy: every action (including ascend/descend)
               comes from a forward pass of the trained network on each
               drone's local observation. This is what the web app runs.

HybridPPOBFS -- the "switch to BFS once damage is found" idea. It uses the
               learned PPO policy to SEARCH, but the moment a drone is near a
               confirmed-damaged region it switches that individual drone into
               a deterministic MEASURE mode: descend to confirmation altitude
               and sweep the frontier of the damaged region (a BFS-style
               boundary expansion) to measure the region's full area. Once the
               region is fully bounded, the drone hands control back to PPO and
               resumes searching. This gives the best of both -- a learned,
               generalizing searcher, plus a systematic, guaranteed-complete
               area survey where it matters.

Both expose .actions(env) and .last_reason, matching every other policy.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from swarm3d import N_ACTIONS

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "ppo_swarm3d.pt")

# action indices (must match swarm3d.ACTION_DELTAS)
N, S, W, E, ASCEND, DESCEND, STAY = range(7)
_ACTION_NAMES = ["north", "south", "west", "east", "ascend", "descend", "stay"]


class OriginalActorCritic(nn.Module):
    """Original network architecture that matches the saved model."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.trunk(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


def _load_net(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained 3D model at {model_path}. Train one with:\n"
            f"  PYTHONPATH=./swarm-env ./swarm-env/bin/python3.14 rl_train3d.py"
        )
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Use original architecture for compatibility with saved model
    net = OriginalActorCritic(ckpt["obs_dim"], ckpt["n_actions"], hidden=ckpt["config"]["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


class PPOPolicy3D:
    def __init__(self, model_path=DEFAULT_MODEL, deterministic=False):
        self.net, self.ckpt = _load_net(model_path)
        self.deterministic = deterministic
        self.last_reason = {}

    def actions(self, env):
        out = {}
        for agent in env.agents:
            ob = torch.tensor(env._get_obs(agent), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.net(ob)
                probs = torch.softmax(logits, dim=-1)[0]
                a = int(torch.argmax(probs).item() if self.deterministic
                        else Categorical(probs=probs).sample().item())
            out[agent] = a
            _, _, alt = env.positions[agent]
            self.last_reason[agent] = (
                f"PPO (learned): {_ACTION_NAMES[a]} @ {probs[a].item():.0%} conf "
                f"| altitude {alt}"
            )
        return out


class HybridPPOBFS:
    """PPO to search, BFS-style frontier sweep to measure a found region."""

    def __init__(self, model_path=DEFAULT_MODEL, switch_radius=2, deterministic=False):
        self.net, self.ckpt = _load_net(model_path)
        self.switch_radius = switch_radius
        self.deterministic = deterministic
        self.last_reason = {}
        self._mode = {}   # agent -> "search" | "measure"

    # ---- helpers -----------------------------------------------------------
    def _ppo_action(self, env, agent):
        ob = torch.tensor(env._get_obs(agent), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(ob)
            probs = torch.softmax(logits, dim=-1)[0]
            return int(torch.argmax(probs).item() if self.deterministic
                       else Categorical(probs=probs).sample().item())

    def _nearby_detected(self, env, r, c):
        """Is there a confirmed-damaged cell within switch_radius?"""
        R = self.switch_radius
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                rr, cc = r + dr, c + dc
                if (0 <= rr < env.grid_size and 0 <= cc < env.grid_size
                        and env.detected_mask[rr, cc]):
                    return True
        return False

    def _measure_frontier_target(self, env, r, c):
        """Nearest UNOBSERVED cell that borders a confirmed-damaged cell --
        the boundary of the known damage region. Visiting it (at low altitude)
        grows the measured area. Returns None if the region is fully bounded
        (nothing left to reveal), meaning measurement is complete."""
        best, best_d = None, None
        gs = env.grid_size
        dmg = env.detected_mask
        for rr in range(gs):
            for cc in range(gs):
                if env.observed[rr, cc]:
                    continue  # already revealed
                # is it adjacent to a confirmed-damaged cell?
                borders = any(
                    0 <= rr + dr < gs and 0 <= cc + dc < gs and dmg[rr + dr, cc + dc]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                if not borders:
                    continue
                d = abs(rr - r) + abs(cc - c)
                if best_d is None or d < best_d:
                    best_d, best = d, (rr, cc)
        return best

    def _step_toward(self, env, agent, target):
        """One legal action moving `agent` toward `target` at low altitude:
        descend to confirmation altitude first, then move horizontally,
        ascending only if terrain blocks the step."""
        r, c, a = env.positions[agent]
        # 1. get low enough to confirm damage (altitude <= 1)
        if a > 1:
            return DESCEND
        # 2. choose the axis with the larger remaining distance
        dr, dc = target[0] - r, target[1] - c
        if abs(dr) >= abs(dc) and dr != 0:
            nr, nc, act = r + np.sign(dr), c, (S if dr > 0 else N)
        elif dc != 0:
            nr, nc, act = r, c + np.sign(dc), (E if dc > 0 else W)
        else:
            return STAY
        nr, nc = int(nr), int(nc)
        # 3. if terrain blocks the horizontal step, climb to clear it
        if a <= env.H[nr, nc]:
            return ASCEND
        return int(act)

    def actions(self, env):
        out = {}
        for agent in env.agents:
            r, c, a = env.positions[agent]
            mode = self._mode.get(agent, "search")

            # enter measure mode when a confirmed-damaged region is close
            if mode == "search" and self._nearby_detected(env, r, c):
                mode = "measure"

            if mode == "measure":
                target = self._measure_frontier_target(env, r, c)
                if target is None:
                    # region fully surveyed -> back to learned search
                    mode = "search"
                    out[agent] = self._ppo_action(env, agent)
                    self.last_reason[agent] = "BFS sweep complete -> back to PPO search"
                else:
                    out[agent] = self._step_toward(env, agent, target)
                    self.last_reason[agent] = (
                        f"BFS area sweep: measuring damage region "
                        f"(frontier {target}, altitude {a})"
                    )
            else:
                out[agent] = self._ppo_action(env, agent)
                self.last_reason[agent] = f"PPO (learned) search | altitude {a}"

            self._mode[agent] = mode
        return out
