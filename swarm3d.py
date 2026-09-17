"""
SwarmSearch3DEnv -- a 3D, altitude-aware disaster-search environment whose
whole point is to be SEARCHED BY A LEARNED (PPO) POLICY.

This is the redesigned core. Unlike the earlier 2D coverage env, here:

  * The world has a THIRD DIMENSION: a terrain height map H(r,c) (buildings,
    mountains). There are no "walls that block" -- instead, tall terrain is
    something a drone must GAIN ALTITUDE to fly over.
  * Each drone has state (row, col, altitude). It must stay above the terrain
    directly beneath it, and to move horizontally onto a taller cell it must
    first climb above that cell's rooftop. That is how obstacle-avoidance is
    expressed: not "can't enter", but "must be high enough".
  * The drone should FLY AS LOW AS POSSIBLE. Low altitude is rewarded (energy
    + sensing), so the learned policy hugs the ground over open areas and only
    climbs to clear buildings. This behaviour is LEARNED, not scripted.
  * Sensing is LIDAR-style and altitude-dependent: from higher up a drone sees
    a WIDER footprint but with LOWER confidence; to CONFIRM damage it must
    descend. This forces the fly-low behaviour to matter for the mission.
  * DAMAGE DETECTION is by comparison: each ground cell has an expected
    signature; a sensed cell whose actual signature differs beyond a
    threshold is flagged as damage (severity 1-3). Detection only succeeds if
    the drone's altitude gives enough confidence.
  * AREA MEASUREMENT: detected damage cells form connected components; the
    measured damaged area is the size of a component. The swarm is rewarded
    for measuring more of a damaged region -> it must EXPLORE MORE around a
    detection, not just flag one cell.
  * ALARM: once a component's measured area crosses a threshold (or a
    "destroyed" cell is confirmed) an alarm fires -> "dispatch rescue".
  * REDISTRIBUTION IS BAKED INTO THE POLICY, two ways: (a) the shared team
    reward means every drone gains when the team measures more damage, and
    (b) each drone's observation includes a "damage beacon" -- a direction +
    proximity to the nearest known, not-yet-fully-measured damage -- so the
    LEARNED policy can route teammates toward a discovery.

All the formulas referenced here are written out in the accompanying PDF.

The class deliberately mirrors the small API the PPO trainer needs:
    .agents, .reset(seed), .step(actions) -> (obs, rewards, terms, truncs, infos),
    ._get_obs(agent), .obs_dim, .n_actions, .action_space(agent).sample()
so it drops into the same training machinery as the 2D env.
"""
import numpy as np
from collections import deque
from gymnasium.spaces import Box, Discrete


# Action indices. Horizontal moves keep altitude; ASCEND/DESCEND change it.
#   0 N, 1 S, 2 W, 3 E, 4 ASCEND, 5 DESCEND, 6 STAY
ACTION_DELTAS = {
    0: (-1, 0, 0), 1: (1, 0, 0), 2: (0, -1, 0), 3: (0, 1, 0),
    4: (0, 0, +1), 5: (0, 0, -1), 6: (0, 0, 0),
}
N_ACTIONS = 7

# Damage severity scale (matches the 2D damage module for consistency).
SEVERITY_LABELS = {0: "intact", 1: "minor", 2: "major", 3: "destroyed"}


class SwarmSearch3DEnv:
    def __init__(self, grid_size=10, n_drones=3, max_steps=60,
                 H_MAX=3, A_MAX=4, obs_window=2,
                 n_buildings=6, n_epicenters=2,
                 detect_conf_thresh=0.75, alarm_area=6, seed=None,
                 # reward weights (documented in the PDF's reward section).
                 # w_alt is deliberately strong so flying LOW is clearly worth
                 # it, and detect_conf_thresh=0.75 means damage can only be
                 # confirmed at altitude <=1 -- so "fly low" and "find damage"
                 # are tightly coupled, giving PPO a clear gradient to learn.
                 w_cov=0.02, w_dmg=1.5, w_area=0.6, w_alt=0.12,
                 w_step=0.01, w_block=0.15, w_alarm=5.0):
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.H_MAX = H_MAX          # tallest terrain
        self.A_MAX = A_MAX          # highest flyable altitude (> H_MAX)
        self.obs_window = obs_window
        self.n_buildings = n_buildings
        self.n_epicenters = n_epicenters
        self.detect_conf_thresh = detect_conf_thresh
        self.alarm_area = alarm_area
        self.rng = np.random.default_rng(seed)

        self.w_cov, self.w_dmg, self.w_area = w_cov, w_dmg, w_area
        self.w_alt, self.w_step, self.w_block = w_alt, w_step, w_block
        self.w_alarm = w_alarm

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.agents = self.possible_agents[:]

        # observation layout (see _get_obs): three (2w+1)^2 patches
        # (terrain, damage, teammates) + own (r,c,a) + confidence + beacon(3)
        p = (2 * obs_window + 1) ** 2
        self.obs_dim = 3 * p + 3 + 1 + 3
        self.n_actions = N_ACTIONS

    # ---- gym-ish helpers so the trainer can sample random actions ----------
    def action_space(self, agent=None):
        return Discrete(N_ACTIONS)

    def observation_space(self, agent=None):
        return Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

    # ---- confidence / footprint models (altitude-dependent LIDAR) ----------
    def sense_confidence(self, altitude):
        """Detection confidence as a function of altitude:
            conf(a) = clip(1 - a / (A_MAX + 1), conf_min, 1)
        Lower altitude -> higher confidence. Damage is only *confirmed* when
        conf(a) >= detect_conf_thresh, so the drone must fly low to confirm."""
        return float(np.clip(1.0 - altitude / (self.A_MAX + 1.0), 0.05, 1.0))

    def footprint_radius(self, altitude):
        """Sensing footprint radius grows with altitude:
            R(a) = R0 + floor(k * a),  here R0 = 1, k = 1.
        Higher = sees wider (good for coverage) but lower confidence (bad for
        confirming damage) -- the core altitude trade-off."""
        return 1 + int(altitude)

    # ---- world generation --------------------------------------------------
    def _generate_terrain(self):
        H = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for _ in range(self.n_buildings):
            h = int(self.rng.integers(1, self.H_MAX + 1))
            bw = int(self.rng.integers(1, 3))
            bh = int(self.rng.integers(1, 3))
            r0 = int(self.rng.integers(0, self.grid_size - bh + 1))
            c0 = int(self.rng.integers(0, self.grid_size - bw + 1))
            H[r0:r0 + bh, c0:c0 + bw] = np.maximum(H[r0:r0 + bh, c0:c0 + bw], h)
        return H

    def _generate_damage(self):
        """Ground-truth damage severity per cell, clustered near random
        epicenters (proximity -> severity), same idea as the 2D damage model.
        'expected' signature is 0 (intact) everywhere; 'actual' = severity."""
        sev = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        eps = [(int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)))
               for _ in range(self.n_epicenters)]
        radius = max(2.0, self.grid_size / 4.0)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                d = min(np.hypot(r - er, c - ec) for er, ec in eps)
                if d <= radius * 0.35:
                    sev[r, c] = 3
                elif d <= radius * 0.6:
                    sev[r, c] = 2
                elif d <= radius:
                    sev[r, c] = 1
        self._epicenters = eps
        return sev

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.steps = 0

        self.H = self._generate_terrain()                 # terrain heights
        self.expected = np.zeros_like(self.H)             # expected signature
        self.actual_damage = self._generate_damage()      # ground-truth severity
        
        # Add obstacles attribute for compatibility with 2D environment
        # In 3D, obstacles are cells with terrain height > 0 (impassable terrain)
        self.obstacles = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.H[r, c] > 0:  # terrain height > 0 means obstacle
                    self.obstacles.add((r, c))

        # Add covered attribute for compatibility with 2D environment
        # In 3D, we track observed cells as "covered"
        self.covered = set()

        # swarm knowledge (what has been sensed / detected so far)
        self.observed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.known_H = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)
        self.detected = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)  # detected severity
        self.detected_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.alarms = []          # list of {"cell","area","severity"} events
        self._alarmed_components = 0

        # place drones at open, low cells, starting high (they must learn to
        # descend). altitude starts at A_MAX.
        self.positions = {}
        free = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                if self.H[r, c] == 0]
        self.rng.shuffle(free)
        for i, agent in enumerate(self.agents):
            r, c = free[i % len(free)]
            self.positions[agent] = (r, c, self.A_MAX)
            self.covered.add((r, c))  # Add starting positions to covered

        self._sense_all()
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: self._info() for a in self.agents}
        return obs, infos

    # ---- sensing -----------------------------------------------------------
    def _sense_all(self):
        """Every drone senses its altitude-dependent footprint. Returns
        (new_observed_count, newly_detected_list) for reward computation."""
        new_obs = 0
        newly_detected = []
        for agent, (r0, c0, a) in self.positions.items():
            R = self.footprint_radius(a)
            conf = self.sense_confidence(a)
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    if dr * dr + dc * dc > R * R:
                        continue
                    r, c = r0 + dr, c0 + dc
                    if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                        continue
                    if not self.observed[r, c]:
                        self.observed[r, c] = True
                        self.covered.add((r, c))  # Update covered set for compatibility
                        self.known_H[r, c] = self.H[r, c]
                        new_obs += 1
                    else:
                        self.known_H[r, c] = self.H[r, c]
                    # DAMAGE BY COMPARISON: |actual - expected| > 0 means
                    # damage; only CONFIRMED if confidence is high enough.
                    sev = int(self.actual_damage[r, c])
                    mismatch = abs(sev - int(self.expected[r, c]))
                    if mismatch > 0 and conf >= self.detect_conf_thresh:
                        if not self.detected_mask[r, c]:
                            self.detected_mask[r, c] = True
                            self.detected[r, c] = sev
                            newly_detected.append((r, c, sev))
        return new_obs, newly_detected

    # ---- damage-area measurement + alarm -----------------------------------
    def _component_area(self, cell):
        """4-connected size of the DETECTED damaged region containing `cell`.
        This is the 'measured area' -- it grows only as the swarm detects more
        of the same contiguous region, which requires flying over and
        confirming it."""
        if not self.detected_mask[cell]:
            return 0, []
        seen = {cell}
        dq = deque([cell])
        comp = [cell]
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + dr, c + dc)
                if (0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size
                        and n not in seen and self.detected_mask[n]):
                    seen.add(n); dq.append(n); comp.append(n)
        return len(comp), comp

    def _check_alarms(self, newly_detected):
        """Fire an alarm when a detected component first reaches alarm_area
        cells, or when a 'destroyed' (sev 3) cell is confirmed."""
        fired = 0
        checked = set()
        for (r, c, sev) in newly_detected:
            if (r, c) in checked:
                continue
            area, comp = self._component_area((r, c))
            checked.update(comp)
            max_sev = max(int(self.detected[x, y]) for x, y in comp)
            if area >= self.alarm_area or max_sev >= 3:
                # only alarm once per growing component (track by a rep cell)
                rep = min(comp)
                if rep not in getattr(self, "_alarm_reps", set()):
                    if not hasattr(self, "_alarm_reps"):
                        self._alarm_reps = set()
                    self._alarm_reps.add(rep)
                    self.alarms.append({"cell": list(rep), "area": area,
                                         "severity": max_sev})
                    fired += 1
        return fired

    # ---- dynamics ----------------------------------------------------------
    def step(self, actions):
        blocked_total = 0
        for agent, act in actions.items():
            r, c, a = self.positions[agent]
            dr, dc, da = ACTION_DELTAS[int(act)]
            nr, nc, na = r + dr, c + dc, a + da
            na = int(np.clip(na, 0, self.A_MAX))

            if da != 0:  # vertical move: clamp so we never go below terrain
                na = max(na, int(self.H[r, c]))  # can't descend into ground
                self.positions[agent] = (r, c, na)
                continue

            # horizontal move: legal only if in bounds AND the drone is above
            # the destination cell's rooftop (must climb tall terrain first)
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                blocked_total += 1
                continue
            if a <= self.H[nr, nc]:
                # too low to clear that terrain -> blocked (must ASCEND first)
                blocked_total += 1
                continue
            self.positions[agent] = (nr, nc, a)

        new_obs, newly_detected = self._sense_all()
        fired = self._check_alarms(newly_detected)

        # ---- reward (shared team reward; see PDF reward section) ----
        dmg_gain = sum(sev for _, _, sev in newly_detected)          # severity-weighted
        area_gain = len(newly_detected)                              # cells newly measured
        mean_alt = np.mean([p[2] for p in self.positions.values()])
        R = (self.w_cov * new_obs
             + self.w_dmg * dmg_gain
             + self.w_area * area_gain
             - self.w_alt * mean_alt
             - self.w_step
             - self.w_block * blocked_total
             + self.w_alarm * fired)

        self.steps += 1
        truncated = self.steps >= self.max_steps
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: float(R) for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: truncated for a in self.agents}
        infos = {a: self._info() for a in self.agents}
        if truncated:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    # ---- observation -------------------------------------------------------
    def _nearest_known_damage_beacon(self, pos):
        """Direction + proximity to the nearest DETECTED damaged cell -- the
        'redistribution beacon'. Returns (unit_dr, unit_dc, inv_dist) so the
        learned policy can steer teammates toward a discovery. Zeros if no
        damage detected yet."""
        r0, c0, _ = pos
        best = None
        best_d = None
        idxs = np.argwhere(self.detected_mask)
        for (r, c) in idxs:
            d = (r - r0) ** 2 + (c - c0) ** 2
            if best_d is None or d < best_d:
                best_d, best = d, (r, c)
        if best is None:
            return (0.0, 0.0, 0.0)
        dr, dc = best[0] - r0, best[1] - c0
        dist = np.hypot(dr, dc)
        if dist < 1e-6:
            return (0.0, 0.0, 1.0)
        return (dr / dist, dc / dist, 1.0 / (1.0 + dist))

    def _get_obs(self, agent):
        r0, c0, a = self.positions[agent]
        w = self.obs_window
        gs = self.grid_size
        terrain_patch, damage_patch, team_patch = [], [], []
        team_cells = {(p[0], p[1]) for ag, p in self.positions.items() if ag != agent}
        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < gs and 0 <= c < gs):
                    terrain_patch.append(-1.0)
                    damage_patch.append(-1.0)
                    team_patch.append(0.0)
                    continue
                # terrain height, normalized; -1 if never sensed
                terrain_patch.append(self.known_H[r, c] / self.H_MAX
                                     if self.observed[r, c] else -1.0)
                # detected damage severity normalized; -1 if unknown/unsensed
                if self.detected_mask[r, c]:
                    damage_patch.append(self.detected[r, c] / 3.0)
                elif self.observed[r, c]:
                    damage_patch.append(0.0)   # sensed, no damage found
                else:
                    damage_patch.append(-1.0)  # not yet sensed
                team_patch.append(1.0 if (r, c) in team_cells else 0.0)
        beacon = self._nearest_known_damage_beacon((r0, c0, a))
        own = [r0 / gs, c0 / gs, a / self.A_MAX]
        conf = [self.sense_confidence(a)]
        return np.array(terrain_patch + damage_patch + team_patch + own + conf + list(beacon),
                        dtype=np.float32)

    # ---- reporting ---------------------------------------------------------
    def _info(self):
        total = self.grid_size * self.grid_size
        return {
            "explored": float(self.observed.sum()) / total,
            "damage_detected": int(self.detected_mask.sum()),
            "damage_total": int((self.actual_damage > 0).sum()),
            "alarms": len(self.alarms),
            "mean_altitude": float(np.mean([p[2] for p in self.positions.values()]))
                             if self.positions else 0.0,
        }
