"""
Live web server for the swarm search simulation.

Runs SwarmCoverageEnv continuously in the background (using the BFS
coverage policy -- see movement_policy.py) and streams drone positions,
battery, and coverage progress to any connected browser over a WebSocket.
The browser renders it on a real Leaflet map (OpenStreetMap tiles, no API
key needed) instead of the matplotlib window used by visualize_swarm.py.

This file doesn't replace visualize_swarm.py -- it's a second, web-based
way to view the exact same simulation, useful for a nicer-looking live demo
in a browser tab instead of a desktop plot window.

Run with:
    PYTHONPATH=./swarm-env ./swarm-env/bin/python3.14 -m uvicorn server:app --reload
Then open http://127.0.0.1:8000 in a browser.

SECURITY NOTE: this starts an unauthenticated local web server. It's fine
for local development/demos on your own machine, but don't expose this
port to the public internet as-is -- there's no login, and anyone who can
reach it can watch (and, if you extend it, potentially control) the sim.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-env"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from swarm import SwarmCoverageEnv
from movement_policy import BFSCoveragePolicy
from unknown_terrain import UnknownTerrainPolicy, FREE, OCCUPIED
from damage import (DamageModel, priority_density, top_priority_zones,
                    SEVERITY_LABELS)

# Two operating MODES, each with its own set of selectable sites:
#
#   known_map      -- the swarm is handed an accurate map up front (real
#                     OpenStreetMap obstacles) and plans routes around known
#                     obstacles with the BFS coverage policy. This is the
#                     original behaviour, unchanged.
#
#   unknown_terrain -- NO map is given. Each drone discovers obstacles live
#                     from limited-range sensing and remembers them in a
#                     shared occupancy grid (see unknown_terrain.py). This is
#                     the search-and-rescue-realistic version, demoed over
#                     real places that have actually experienced disasters.
#
# All sites are real named places used as stand-in search areas; there is no
# live incident data feed. bbox = (west, south, east, north) lon/lat degrees.
KNOWN_LOCATIONS = {
    "san_jose": {
        "label": "Downtown San Jose, CA",
        "bbox": (-121.94, 37.32, -121.87, 37.37),
        "grid_size": 25,
    },
    "berkeley": {
        "label": "UC Berkeley campus area",
        "bbox": (-122.259, 37.870, -122.253, 37.875),
        "grid_size": 20,
    },
    "golden_gate_park": {
        "label": "Golden Gate Park, SF",
        "bbox": (-122.511, 37.765, -122.454, 37.775),
        "grid_size": 25,
    },
}

DISASTER_LOCATIONS = {
    "amatrice": {
        "label": "Amatrice, Italy (2016 earthquake)",
        "bbox": (13.280, 42.620, 13.310, 42.640),
        "grid_size": 20,
    },
    "lahaina": {
        "label": "Lahaina, Maui (2023 wildfire)",
        "bbox": (-156.695, 20.868, -156.665, 20.892),
        "grid_size": 22,
    },
    "christchurch": {
        "label": "Christchurch, NZ (2011 earthquake)",
        "bbox": (172.620, -43.540, 172.645, -43.525),
        "grid_size": 22,
    },
    "kahramanmaras": {
        "label": "Kahramanmaras, Turkey (2023 earthquake)",
        "bbox": (36.910, 37.565, 36.935, 37.585),
        "grid_size": 22,
    },
}

MODES = {
    "known_map": {
        "label": "Known map -- plan around known obstacles",
        "locations": KNOWN_LOCATIONS,
        "default_location": "san_jose",
    },
    "unknown_terrain": {
        "label": "Unknown terrain -- discover obstacles live (disaster zones)",
        "locations": DISASTER_LOCATIONS,
        "default_location": "amatrice",
    },
    "ppo_3d": {
        # THE LEARNED policy running live: a PPO network (trained by
        # rl_train3d.py, weights in models/ppo_swarm3d.pt) flies a 3D,
        # altitude-aware search -- climbing over terrain, diving low to
        # confirm damage, and raising alarms. Terrain/damage are synthetic
        # but placed over a real disaster bbox for display.
        "label": "3D learned search (PPO) -- altitude + damage detection",
        "locations": DISASTER_LOCATIONS,
        "default_location": "amatrice",
    },
}

DEFAULT_MODE = "known_map"
SENSOR_RADIUS = 3  # drone sensing range (cells) in unknown_terrain mode
MIN_DRONES = 1
MAX_DRONES = 10
DEFAULT_N_DRONES = 4

MAX_STEPS = 150
PPO3D_GRID = 14        # must sit in the trained model's distribution
PPO3D_STEPS = 120      # longer episodes than training, for a smoother demo
PPO3D_BUILDINGS = 10

# The trained 3D policy is stateless across episodes, so load it once and
# reuse it (avoids re-reading the weights file on every reset).
_ppo3d_policy = None


def get_ppo3d_policy():
    global _ppo3d_policy
    if _ppo3d_policy is None:
        from rl_policy3d import PPOPolicy3D
        _ppo3d_policy = PPOPolicy3D()
    return _ppo3d_policy
TICK_SECONDS = 0.25  # simulated time between broadcasts -- lower = faster playback

# Grid movement delta -> real-world compass bearing in degrees (clockwise
# from north), matching how a map marker's rotation is normally expressed.
# NOTE: this is a different convention from HEADING_ANGLES in
# visualize_swarm.py, which is tuned for matplotlib's on-screen axes -- map
# bearings and screen-rotation angles aren't the same thing.
HEADING_BEARINGS = {
    (-1, 0): 0,     # moved up/north
    (1, 0): 180,    # moved down/south
    (0, -1): 270,   # moved left/west
    (0, 1): 90,     # moved right/east
}

app = FastAPI()


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts JSON to all of
    them, quietly dropping any that have disconnected."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


class SimulationRunner:
    """Owns one long-running SwarmCoverageEnv episode loop. When an episode
    truncates, it resets and keeps going -- this is meant to run forever as
    a live demo, not a single one-shot script.

    Also supports switching location or drone count live: rebuild_env()
    tears down the current env and builds a fresh one, which necessarily
    means a full episode reset (obstacles, coverage, and battery for the
    new area/swarm size don't carry over from the old one -- there's no
    sensible way to "resume" a search of one place with a swarm sized for
    a different one).
    """

    def __init__(self, mode=DEFAULT_MODE, location_key=None, n_drones=DEFAULT_N_DRONES):
        self.mode = None
        self.location_key = None
        self.n_drones = None
        self.env = None
        self.policy = None
        self.headings = {}
        self._prev_occ = None  # snapshot of occupancy grid for tick-diffing
        # When True (unknown_terrain only), the swarm biases its routing
        # toward discovered high-severity damage. Toggle off to see the
        # "before" behaviour -- pure exploration ignoring damage -- so you
        # can compare how the drones fly with vs without prioritization.
        self.priority_routing = True
        # guards the sim loop from reading self.env mid-rebuild
        self.lock = asyncio.Lock()
        self.rebuild_env(mode, location_key, n_drones)

    def rebuild_env(self, mode, location_key, n_drones):
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        locations = MODES[mode]["locations"]
        if location_key is None:
            location_key = MODES[mode]["default_location"]
        if location_key not in locations:
            raise ValueError(f"unknown location {location_key!r} for mode {mode!r}")
        n_drones = max(MIN_DRONES, min(MAX_DRONES, int(n_drones)))

        loc = locations[location_key]
        self.mode = mode
        self.location_key = location_key
        self.n_drones = n_drones
        self._display_bbox = loc["bbox"]

        if mode == "ppo_3d":
            from swarm3d import SwarmSearch3DEnv
            self.env = SwarmSearch3DEnv(
                grid_size=PPO3D_GRID, n_drones=n_drones, max_steps=PPO3D_STEPS,
                n_buildings=PPO3D_BUILDINGS,
            )
        else:
            self.env = SwarmCoverageEnv(
                grid_size=loc["grid_size"], n_drones=n_drones, max_steps=MAX_STEPS,
                bbox=loc["bbox"],
            )
        self._reset_episode()

    def _reset_episode(self):
        self.env.reset()
        if self.mode == "ppo_3d":
            # reuse the singleton trained policy (stateless across episodes)
            self.policy = get_ppo3d_policy()
            self._prev_detected_mask = self.env.detected_mask.copy()
            self._prev_alarms = 0
            self.headings = {agent: 0 for agent in self.env.agents}
            return
        if self.mode == "unknown_terrain":
            self.policy = UnknownTerrainPolicy(sensor_radius=SENSOR_RADIUS)
            self.policy.reset(self.env)
            self._prev_occ = self.policy.occ.state.copy()
            # ground-truth damage for this disaster site (synthetic stand-in
            # for a real damage classifier -- see damage.py). Drones only
            # LEARN a cell's severity once they sense it; self.sensed_damage
            # is the discovered-so-far subset, exactly like the occupancy
            # grid is discovered-so-far obstacles.
            self.damage_model = DamageModel(
                self.env.grid_size, self.env.obstacles,
                seed=self.env.rng.integers(1_000_000), n_epicenters=2,
            )
            self.sensed_damage = {}  # (r,c) -> severity, only where sensed
        else:
            self.policy = BFSCoveragePolicy()
            self._prev_occ = None
            self.damage_model = None
            self.sensed_damage = {}
        self.headings = {agent: 0 for agent in self.env.agents}

    def _priority_grid(self):
        """Gaussian priority-density grid built from damage discovered so
        far (0 everywhere not-yet-sensed-as-damaged). Used both to bias the
        swarm's routing and to rank zones for the UI."""
        sev = np.zeros((self.env.grid_size, self.env.grid_size), dtype=float)
        for (r, c), s in self.sensed_damage.items():
            sev[r, c] = s
        if not self.sensed_damage:
            return None
        return priority_density(sev, sigma=2.0)

    def _damage_counts(self):
        """Running tally of discovered damage by severity label, for the UI."""
        counts = {label: 0 for label in SEVERITY_LABELS.values() if label != "intact"}
        for s in self.sensed_damage.values():
            counts[SEVERITY_LABELS[s]] += 1
        return counts

    def inflict_damage(self, lat, lon, severity=3, snap_radius=2):
        """Interactive damage tool: damage the structure at a clicked
        lat/lon. Snaps to the clicked cell if it's a structure, else to the
        nearest structure within snap_radius cells (clicks rarely land dead-
        centre on a building). Only mutates ground truth -- the swarm won't
        'know' until a drone senses it. Returns info for UI feedback, or
        None if the click wasn't on/near any structure."""
        if self.mode != "unknown_terrain" or self.damage_model is None:
            return None
        cell = self.env.latlon_to_cell(lat, lon)
        if cell is None:
            return None

        target = None
        if cell in self.env.obstacles:
            target = cell
        else:
            # snap to the nearest structure within a small radius
            best_d = None
            for (r, c) in self.env.obstacles:
                d = abs(r - cell[0]) + abs(c - cell[1])
                if d <= snap_radius and (best_d is None or d < best_d):
                    best_d, target = d, (r, c)
        if target is None:
            return None

        new_sev = self.damage_model.inflict(target, severity)
        r, c = target
        already_seen = target in self.sensed_damage
        return {
            "cell": [r, c],
            "bounds": list(self.env._cell_bounds(r, c)),
            "severity": new_sev,
            # whether a drone currently has eyes on it (so the user knows if
            # recognition will be immediate or pending a fly-by)
            "in_view": target in getattr(self.policy, "last_visible_occupied", set()),
            "already_detected": already_seen,
        }

    def _locations(self):
        return MODES[self.mode]["locations"]

    def location_center(self):
        west, south, east, north = self._locations()[self.location_key]["bbox"]
        return {"lat": (south + north) / 2, "lon": (west + east) / 2}

    def obstacles_payload(self):
        """[west, south, east, north] bounds for every obstacle cell. Only
        used in known_map mode -- in unknown_terrain mode obstacles are
        hidden and revealed incrementally through sensing instead."""
        return [list(self.env._cell_bounds(r, c)) for (r, c) in self.env.obstacles]

    def drone_payload(self):
        payload = {}
        reasons = getattr(self.policy, "last_reason", {})
        for agent, pos in self.env.positions.items():
            lat, lon = self.env.cell_to_latlon(*pos)
            payload[agent] = {
                "lat": lat,
                "lon": lon,
                "heading": self.headings.get(agent, 0),
                "battery": round(self.env.battery[agent], 1),
                "status": reasons.get(agent, ""),
            }
        return payload

    # ---- 3D (ppo_3d) display helpers --------------------------------------
    def cell3d_bounds(self, r, c):
        """Map a 3D-grid cell to lon/lat bounds over the chosen display bbox
        (the 3D env is synthetic and has no geography of its own)."""
        west, south, east, north = self._display_bbox
        gs = self.env.grid_size
        lon_step = (east - west) / gs
        lat_step = (north - south) / gs
        cell_north = north - r * lat_step
        cell_south = cell_north - lat_step
        cell_west = west + c * lon_step
        cell_east = cell_west + lon_step
        return (cell_west, cell_south, cell_east, cell_north)

    def cell3d_latlon(self, r, c):
        w, s, e, n = self.cell3d_bounds(r, c)
        return ((s + n) / 2, (w + e) / 2)

    def terrain_payload_3d(self):
        """Every terrain cell with height > 0, as [w,s,e,n,height], so the
        frontend can shade buildings/mountains (taller = darker)."""
        out = []
        gs = self.env.grid_size
        for r in range(gs):
            for c in range(gs):
                h = int(self.env.H[r, c])
                if h > 0:
                    out.append(list(self.cell3d_bounds(r, c)) + [h])
        return out

    def drone_payload_3d(self):
        payload = {}
        reasons = getattr(self.policy, "last_reason", {})
        for agent, (r, c, a) in self.env.positions.items():
            lat, lon = self.cell3d_latlon(r, c)
            payload[agent] = {
                "lat": lat, "lon": lon,
                "altitude": int(a),
                "alt_frac": a / self.env.A_MAX,   # 0 (ground) .. 1 (ceiling)
                "heading": self.headings.get(agent, 0),
                "status": reasons.get(agent, ""),
            }
        return payload

    def _damage_counts_3d(self):
        counts = {label: 0 for label in SEVERITY_LABELS.values() if label != "intact"}
        for (r, c) in zip(*self.env.detected_mask.nonzero()):
            counts[SEVERITY_LABELS[int(self.env.detected[r, c])]] += 1
        return counts

    def _step_3d(self):
        """One tick of the learned 3D search. Streams altitude, terrain,
        newly-detected damage, and alarms."""
        prev_positions = dict(self.env.positions)
        actions = self.policy.actions(self.env)
        _, rewards, _, truncs, infos = self.env.step(actions)

        # heading from horizontal movement only (ascend/descend don't turn)
        for agent, (r, c, a) in self.env.positions.items():
            pr, pc, pa = prev_positions[agent]
            d = (r - pr, c - pc)
            if d in HEADING_BEARINGS:
                self.headings[agent] = HEADING_BEARINGS[d]

        info = list(infos.values())[0] if infos else {}

        # newly CONFIRMED damage cells since last tick -> events with popups
        cur = self.env.detected_mask
        newly = cur & (~self._prev_detected_mask)
        damage_events = []
        for (r, c) in zip(*newly.nonzero()):
            r, c = int(r), int(c)
            sev = int(self.env.detected[r, c])
            lat, lon = self.cell3d_latlon(r, c)
            damage_events.append({
                "bounds": list(self.cell3d_bounds(r, c)), "lat": lat, "lon": lon,
                "severity": sev, "label": SEVERITY_LABELS[sev],
                "description": f"{SEVERITY_LABELS[sev].capitalize()} damage confirmed by drone LIDAR.",
            })
        self._prev_detected_mask = cur.copy()

        # alarms raised since last tick -> "dispatch rescue" events
        new_alarms = []
        for al in self.env.alarms[self._prev_alarms:]:
            ar, ac = al["cell"]
            lat, lon = self.cell3d_latlon(ar, ac)
            new_alarms.append({
                "lat": lat, "lon": lon, "area": al["area"],
                "severity": al["severity"], "label": SEVERITY_LABELS[al["severity"]],
            })
        self._prev_alarms = len(self.env.alarms)

        message = {
            "type": "tick",
            "step": self.env.steps, "max_steps": self.env.max_steps,
            "explored": info.get("explored", 0.0),
            "damage_detected": info.get("damage_detected", 0),
            "damage_total": info.get("damage_total", 0),
            "mean_altitude": round(info.get("mean_altitude", 0.0), 2),
            "alarms_total": info.get("alarms", 0),
            "A_MAX": self.env.A_MAX,
            "drones": self.drone_payload_3d(),
            "damage_events": damage_events,
            "new_alarms": new_alarms,
            "damage_counts": self._damage_counts_3d(),
        }

        just_reset = False
        if not self.env.agents:
            self._reset_episode()
            just_reset = True
        return message, just_reset

    def step(self):
        """Advance one tick. Returns (tick_message, just_reset)."""
        if self.mode == "ppo_3d":
            return self._step_3d()
        prev_positions = self.env.positions
        prev_covered = set(self.env.covered)

        # In unknown_terrain mode the policy senses (updating its occupancy
        # grid) as part of choosing actions. Pass the current damage-priority
        # grid so the swarm biases toward discovered high-severity clusters.
        if self.mode == "unknown_terrain":
            pri = self._priority_grid() if self.priority_routing else None
            actions = self.policy.actions(self.env, priority=pri)
        else:
            actions = self.policy.actions(self.env)
        obs, rewards, terms, truncs, infos = self.env.step(actions)

        for agent, pos in self.env.positions.items():
            prev = prev_positions[agent]
            delta = (pos[0] - prev[0], pos[1] - prev[1])
            if delta in HEADING_BEARINGS:
                self.headings[agent] = HEADING_BEARINGS[delta]

        coverage = list(infos.values())[0]["coverage"] if infos else 0.0

        message = {
            "type": "tick",
            "step": self.env.steps,
            "max_steps": self.env.max_steps,
            "coverage": coverage,
            "drones": self.drone_payload(),
        }

        if self.mode == "unknown_terrain":
            # Report cells whose occupancy state changed since last tick, so
            # the frontend can lift the "fog" incrementally and draw newly
            # discovered obstacles. explored = fraction of the map no longer
            # unknown (the real progress metric in this mode).
            occ = self.policy.occ.state
            newly_free, newly_occ = [], []
            changed = (occ != self._prev_occ)
            for (r, c) in zip(*changed.nonzero()):
                r, c = int(r), int(c)
                bounds = list(self.env._cell_bounds(r, c))
                if occ[r, c] == FREE:
                    newly_free.append(bounds)
                elif occ[r, c] == OCCUPIED:
                    newly_occ.append(bounds)
            self._prev_occ = occ.copy()

            # Damage RE-DETECTION: for every structure a drone can currently
            # SEE this tick (not just newly-sensed ones), run the "damage
            # classifier" (severity_of seam). Emit an event whenever the
            # severity a drone observes is higher than what we'd previously
            # recorded -- this covers both first sight of pre-existing damage
            # AND user-inflicted damage that appears on an already-known
            # building (the drone re-observes it and notices it's now worse).
            # Crucially, a cell the user damaged is NOT recognized until a
            # drone actually has it in view -- that's the "do the drones
            # recognize it?" behaviour working as intended.
            damage_events = []
            for cell in self.policy.last_visible_occupied:
                sev = self.damage_model.severity_of(cell)
                if sev > 0 and sev > self.sensed_damage.get(cell, 0):
                    self.sensed_damage[cell] = sev
                    r, c = cell
                    lat, lon = self.env.cell_to_latlon(r, c)
                    damage_events.append({
                        "bounds": list(self.env._cell_bounds(r, c)),
                        "lat": lat, "lon": lon,
                        "severity": sev,
                        "label": SEVERITY_LABELS[sev],
                        "description": self.damage_model.describe(cell),
                    })

            # rank discovered damage into distinct priority zones for the UI
            zones = []
            pri = self._priority_grid()
            if pri is not None:
                for (zr, zc, score) in top_priority_zones(pri, k=3):
                    lat, lon = self.env.cell_to_latlon(int(zr), int(zc))
                    zones.append({"lat": lat, "lon": lon, "score": round(float(score), 3)})

            message["new_sensed_free"] = newly_free
            message["new_sensed_obstacle"] = newly_occ
            message["explored"] = self.policy.occ.explored_fraction()
            message["damage_events"] = damage_events
            message["damage_counts"] = self._damage_counts()
            message["priority_zones"] = zones
        else:
            new_cells = set(self.env.covered) - prev_covered
            message["new_covered"] = [list(self.env._cell_bounds(r, c)) for (r, c) in new_cells]

        just_reset = False
        if not self.env.agents:  # episode truncated
            self._reset_episode()
            just_reset = True

        return message, just_reset


runner = SimulationRunner()


def init_payload():
    payload = {
        "type": "init",
        "mode": runner.mode,
        "modes": {k: v["label"] for k, v in MODES.items()},
        "location_key": runner.location_key,
        # locations available for the CURRENT mode (frontend swaps the site
        # dropdown when the mode changes)
        "locations": {k: v["label"] for k, v in runner._locations().items()},
        "n_drones": runner.n_drones,
        "min_drones": MIN_DRONES,
        "max_drones": MAX_DRONES,
        "priority_routing": runner.priority_routing,
        "grid_size": runner.env.grid_size,
        # obstacles are only revealed up-front in known_map mode; in
        # unknown_terrain they start hidden and are sensed live
        "obstacles": runner.obstacles_payload() if runner.mode == "known_map" else [],
        "center": runner.location_center(),
        "bbox": list(runner._locations()[runner.location_key]["bbox"]),
    }
    if runner.mode == "ppo_3d":
        # terrain heights to shade, plus altitude ceiling for the drone
        # altitude colour scale; no obstacles/priority in this mode.
        payload["terrain"] = runner.terrain_payload_3d()
        payload["A_MAX"] = runner.env.A_MAX
        payload["H_MAX"] = runner.env.H_MAX
    return payload


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # New client gets the current location/obstacle layout immediately --
    # obstacles are fixed per bbox, so this only needs to happen once per
    # client (or again after a rebuild, see /api/configure below), not
    # every tick.
    async with runner.lock:
        await ws.send_json(init_payload())
    try:
        while True:
            # Keep the connection open; the actual simulation loop
            # (below) is what pushes ticks. We just need to detect
            # disconnects here.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


class ConfigureRequest(BaseModel):
    mode: str
    location: str | None = None
    n_drones: int
    priority_routing: bool | None = None


@app.post("/api/configure")
async def configure(req: ConfigureRequest):
    """Switch the live simulation to a different mode, search site, and/or
    drone count -- the 'radar: pick a mode + zone and send N drones' control.
    Rebuilds the environment (a fresh episode; see rebuild_env's docstring
    for why state can't carry over) and re-broadcasts a fresh init payload
    to every connected browser so their overlay/sidebar update to match the
    new configuration immediately, instead of waiting for the next tick.
    """
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode!r}")
    locations = MODES[req.mode]["locations"]
    # Tolerate a location that doesn't belong to the chosen mode (e.g. the
    # UI still had a known-map site selected when switching to a disaster
    # mode): fall back to this mode's default rather than erroring, so the
    # deploy always succeeds and the fresh init corrects the site dropdown.
    location = req.location
    if location is not None and location not in locations:
        location = None
    async with runner.lock:
        if req.priority_routing is not None:
            runner.priority_routing = bool(req.priority_routing)
        runner.rebuild_env(req.mode, location, req.n_drones)
        await manager.broadcast(init_payload())
        await manager.broadcast({"type": "reset"})
    return {"ok": True, "mode": runner.mode, "location": runner.location_key,
            "n_drones": runner.n_drones, "priority_routing": runner.priority_routing}


class InflictRequest(BaseModel):
    lat: float
    lon: float
    severity: int | None = 3


@app.post("/api/inflict")
async def inflict(req: InflictRequest):
    """Damage the structure at a clicked map location (unknown_terrain mode
    only). The swarm does NOT immediately know -- a drone has to sense the
    cell first, at which point the re-detection loop in step() surfaces it
    as a damage event. Returns the affected cell + whether it's currently in
    a drone's view, so the UI can show a 'pending detection' hint."""
    if runner.mode != "unknown_terrain":
        raise HTTPException(
            status_code=400,
            detail="inflicting damage is only available in unknown_terrain mode",
        )
    sev = 3 if req.severity is None else max(1, min(3, int(req.severity)))
    async with runner.lock:
        result = runner.inflict_damage(req.lat, req.lon, severity=sev)
    if result is None:
        return {"ok": False, "reason": "no structure at or near that point"}
    return {"ok": True, **result}


async def simulation_loop():
    """Runs forever in the background: step the sim, broadcast the result,
    wait, repeat. Starts as soon as the server starts, independent of
    whether any browser is currently connected."""
    while True:
        async with runner.lock:
            message, just_reset = runner.step()
            # ppo_3d regenerates terrain each episode, so on reset re-send a
            # full init (fresh terrain) rather than a bare reset signal.
            reset_is_init = just_reset and runner.mode == "ppo_3d"
            init_msg = init_payload() if reset_is_init else None
        if reset_is_init:
            await manager.broadcast(init_msg)
        elif just_reset:
            await manager.broadcast({"type": "reset"})
        await manager.broadcast(message)
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(simulation_loop())


_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_WEB_DIR, "index.html"), "r") as f:
        return f.read()


app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
