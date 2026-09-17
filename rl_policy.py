"""
Inference wrapper for the trained PPO policy.

PPOPolicy loads the weights saved by rl_train.py and exposes the same
.actions(env) interface as BFSCoveragePolicy / UnknownTerrainPolicy, so a
learned policy is a drop-in replacement for the hand-coded ones anywhere
they're used (e.g. visualize_swarm.py's demo_rollout).

Unlike the hand-coded policies, this one makes no plan and knows no map --
it just runs each drone's local observation through the trained network and
takes the action the network prefers. All the "intelligence" lives in the
learned weights, which is the whole point of the RL version.
"""
import os

import numpy as np
import torch
from torch.distributions import Categorical

from rl_train import ActorCritic

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "ppo_swarm.pt")


class PPOPolicy:
    def __init__(self, model_path=DEFAULT_MODEL, deterministic=False):
        """
        model_path : path to the checkpoint saved by rl_train.py.
        deterministic : SAMPLING MODE. Default False (sample from the policy)
            is the correct way to run this shared multi-agent policy: with
            one shared network, identical drones taking greedy (argmax)
            actions move in lockstep and clump, tanking coverage. Sampling
            breaks that symmetry so drones spread out. (Measured: stochastic
            ~45% coverage vs greedy ~13% on the training task.) Set True only
            if you specifically want the deterministic argmax behavior.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Train one first with:\n"
                f"  PYTHONPATH=./swarm-env ./swarm-env/bin/python3.14 rl_train.py"
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.obs_dim = ckpt["obs_dim"]
        self.n_actions = ckpt["n_actions"]
        self.obs_window = ckpt["config"]["obs_window"]
        self.net = ActorCritic(self.obs_dim, self.n_actions,
                               hidden=ckpt["config"]["hidden"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.deterministic = deterministic
        self.last_reason = {}
        self._action_names = ["up", "down", "left", "right", "stay"]

    def actions(self, env):
        """Return {agent: action} for every active drone by running each
        drone's local observation through the trained network."""
        if env.obs_window != self.obs_window:
            raise ValueError(
                f"model was trained with obs_window={self.obs_window} but env "
                f"has obs_window={env.obs_window}; observation sizes won't match"
            )
        out = {}
        for agent in env.agents:
            ob = torch.tensor(env._get_obs(agent), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = self.net(ob)
                probs = torch.softmax(logits, dim=-1)[0]
                if self.deterministic:
                    a = int(torch.argmax(probs).item())
                else:
                    a = int(Categorical(probs=probs).sample().item())
            out[agent] = a
            self.last_reason[agent] = (
                f"PPO policy (learned): action '{self._action_names[a]}' "
                f"at {probs[a].item():.0%} confidence"
            )
        return out
