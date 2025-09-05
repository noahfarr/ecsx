# SPDX-License-Identifier: MIT
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from ecsx.world.world_api import World
from ecsx.core.component_specification import ComponentSpecification
from ecsx.systems.pipeline_builder import build_step_unrolled
from ecsx.systems.pipeline_batched_static import build_batched_step_static
from ecsx.core.static_world import StaticWorld
from ecsx.utils.tree_batch import stack_trees
from ecsx.query.query_engine import select

# ---- try raylib bindings ----
try:
    import pyray as rl  # pip install raylib-py or pyray
except Exception:  # pragma: no cover
    import raylibpy as rl  # fallback

# --------------------------- Components & Params ---------------------------

def get_grid_specs(obs_dim: int = 7):
    return (
        ComponentSpecification("GridPosition", (2,), jnp.int32, jnp.array([0, 0], jnp.int32)),
        ComponentSpecification("Team", (1,), jnp.int32, jnp.array([0], jnp.int32)),
        ComponentSpecification("CarryingOpponentFlag", (1,), jnp.bool_, jnp.array([False])),
        ComponentSpecification("Observation", (obs_dim,), jnp.float32, jnp.zeros((obs_dim,), jnp.float32)),
        ComponentSpecification("DiscreteAction", (), jnp.int32, jnp.array(0, jnp.int32)),
        ComponentSpecification("Reward", (1,), jnp.float32, jnp.array([0.0], jnp.float32)),
        ComponentSpecification("Termination", (1,), jnp.bool_, jnp.array([False])),
    )

@dataclass(frozen=True)
class CtfParams:
    grid_width: int = 11
    grid_height: int = 7
    team0_base: Tuple[int, int] = (1, 3)
    team1_base: Tuple[int, int] = (9, 3)
    team0_flag: Tuple[int, int] = (1, 3)
    team1_flag: Tuple[int, int] = (9, 3)
    step_penalty: float = -0.00
    end_on_capture: bool = True

# ------------------------------- Systems ---------------------------------------

def system_grid_movement_from_discrete(params: CtfParams):
    W = jnp.array(params.grid_width, jnp.int32)
    H = jnp.array(params.grid_height, jnp.int32)
    LUT = jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32)
    def _sys(world, key, inputs):
        mask = select(world, required=("GridPosition", "DiscreteAction"))
        pos_store = world.component_stores["GridPosition"]
        act_store = world.component_stores["DiscreteAction"]
        pos = pos_store.data
        a = jnp.clip(act_store.data, 0, 4)
        delta = LUT[a]
        new_pos = jnp.stack([jnp.clip(pos[:, 0] + delta[:, 0], 0, W - 1),
                             jnp.clip(pos[:, 1] + delta[:, 1], 0, H - 1)], axis=1)
        upd = jnp.where(mask[:, None], new_pos, pos)
        return world._with_store("GridPosition", type(pos_store)(pos_store.spec, upd, pos_store.alive_mask))
    return _sys

def system_ctf_logic(params: CtfParams):
    base0 = jnp.array(params.team0_base, jnp.int32)
    base1 = jnp.array(params.team1_base, jnp.int32)
    flag0 = jnp.array(params.team0_flag, jnp.int32)
    flag1 = jnp.array(params.team1_flag, jnp.int32)
    step_penalty = jnp.array(params.step_penalty, jnp.float32)

    def _sys(world, key, inputs):
        pos = world.component_stores["GridPosition"].data
        team = world.component_stores["Team"].data[:, 0]
        carry_store = world.component_stores["CarryingOpponentFlag"]
        carry = carry_store.data[:, 0]
        alive = world.alive_mask
        agents = select(world, required=("Team", "GridPosition"))

        # step penalty
        rew_store = world.component_stores["Reward"]
        rew = rew_store.data[:, 0]
        rew = jnp.where(alive, rew + step_penalty, rew)

        opp_flag = jnp.where(team[:, None] == 0, flag1[None, :], flag0[None, :])
        on_opp_flag = jnp.all(pos == opp_flag, axis=-1) & agents
        carry = jnp.where(on_opp_flag, True, carry)

        same = jnp.all(pos[:, None, :] == pos[None, :, :], axis=-1)
        enemy = (team[:, None] != team[None, :])
        collided = jnp.any(same & enemy, axis=1) & agents
        home_base = jnp.where(team[:, None] == 0, base0[None, :], base1[None, :])
        pos_new = jnp.where(collided[:, None], home_base, pos)
        carry = jnp.where(collided, False, carry)

        on_own = jnp.all(pos_new == jnp.where(team[:, None] == 0, base0[None, :], base1[None, :]), axis=-1)
        scored = carry & on_own & agents
        t0_scored = jnp.any(scored & (team == 0))
        t1_scored = jnp.any(scored & (team == 1))
        give0 = (team == 0) & t0_scored
        give1 = (team == 1) & t1_scored
        rew = jnp.where(give0 | give1, rew + 1.0, rew)
        carry = jnp.where(scored, False, carry)

        term_store = world.component_stores["Termination"]
        term = term_store.data[:, 0]
        any_scored = jnp.any(scored)
        term = jnp.where(agents, jnp.where(any_scored, True, term), term)

        world = world._with_store("GridPosition", type(world.component_stores["GridPosition"])(
            world.component_stores["GridPosition"].spec, pos_new, world.component_stores["GridPosition"].alive_mask))
        world = world._with_store("CarryingOpponentFlag", type(carry_store)(
            carry_store.spec, carry_store.data.at[:, 0].set(carry), carry_store.alive_mask))
        world = world._with_store("Reward", type(rew_store)(
            rew_store.spec, rew_store.data.at[:, 0].set(rew), rew_store.alive_mask))
        world = world._with_store("Termination", type(term_store)(
            term_store.spec, term_store.data.at[:, 0].set(term), term_store.alive_mask))
        return world
    return _sys

def system_observation(params: CtfParams):
    W = jnp.array(params.grid_width, jnp.float32)
    H = jnp.array(params.grid_height, jnp.float32)
    base0 = jnp.array(params.team0_base, jnp.float32)
    base1 = jnp.array(params.team1_base, jnp.float32)
    flag0 = jnp.array(params.team0_flag, jnp.float32)
    flag1 = jnp.array(params.team1_flag, jnp.float32)
    def _sys(world, key, inputs):
        mask = select(world, required=("Observation", "GridPosition", "Team", "CarryingOpponentFlag"))
        obs_store = world.component_stores["Observation"]
        pos = world.component_stores["GridPosition"].data.astype(jnp.float32)
        team = world.component_stores["Team"].data[:, 0]
        carry = world.component_stores["CarryingOpponentFlag"].data[:, 0]
        own_base = jnp.where(team[:, None] == 0, base0[None, :], base1[None, :])
        opp_flag = jnp.where(team[:, None] == 0, flag1[None, :], flag0[None, :])
        feats = jnp.concatenate([pos / jnp.array([W, H])[None, :],
                                 own_base / jnp.array([W, H])[None, :],
                                 opp_flag / jnp.array([W, H])[None, :],
                                 carry.astype(jnp.float32)[:, None]], axis=1)
        new = feats[:, :obs_store.spec.shape[0]]
        upd = jnp.where(mask[:, None], new, obs_store.data)
        return world._with_store("Observation", type(obs_store)(obs_store.spec, upd, obs_store.alive_mask))
    return _sys

# ----------------------------- Builders -----------------------------

def build_world(n_per_team: int, params: CtfParams) -> Tuple[World, jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(0)
    w = World(capacity=2 * n_per_team, key=key)
    for spec in get_grid_specs(obs_dim=7):
        w = w.register_component(spec)

    b0x, b0y = params.team0_base; b1x, b1y = params.team1_base
    t0, t1 = [], []
    for _ in range(n_per_team):
        e = w.spawn(GridPosition=jnp.array([b0x, b0y], jnp.int32),
                    Team=jnp.array([0], jnp.int32),
                    CarryingOpponentFlag=jnp.array([False]),
                    Observation=jnp.zeros((7,), jnp.float32),
                    DiscreteAction=jnp.array(0, jnp.int32),
                    Reward=jnp.zeros((1,), jnp.float32),
                    Termination=jnp.array([False]))
        t0.append(e)
    for _ in range(n_per_team):
        e = w.spawn(GridPosition=jnp.array([b1x, b1y], jnp.int32),
                    Team=jnp.array([1], jnp.int32),
                    CarryingOpponentFlag=jnp.array([False]),
                    Observation=jnp.zeros((7,), jnp.float32),
                    DiscreteAction=jnp.array(0, jnp.int32),
                    Reward=jnp.zeros((1,), jnp.float32),
                    Termination=jnp.array([False]))
        t1.append(e)
    return w, jnp.asarray(t0, jnp.int32), jnp.asarray(t1, jnp.int32)

def make_step(params: CtfParams, unroll: int = 1):
    return build_step_unrolled(
        (system_grid_movement_from_discrete(params),
         system_ctf_logic(params),
         system_observation(params)),
        unroll=unroll,
    )

def freeze_batch(B: int, n_per_team: int, params: CtfParams) -> StaticWorld:
    sws = []
    for _ in range(B):
        w, _, _ = build_world(n_per_team, params)
        order = ("GridPosition", "Team", "CarryingOpponentFlag",
                 "Observation", "DiscreteAction", "Reward", "Termination")
        sws.append(StaticWorld.freeze(w.state, order))
    return stack_trees(sws)

# ----------------------------- Renderer -----------------------------

class CtfRenderer:
    def __init__(self, params: CtfParams, cell_px: int = 48):
        self.p = params
        self.cell = cell_px
        w = params.grid_width * cell_px
        h = params.grid_height * cell_px
        rl.init_window(w, h, "CTF Gridworld")
        rl.set_target_fps(60)

    def close(self):
        rl.close_window()

    def _cell_rect(self, x, y):
        return rl.Rectangle(x * self.cell, y * self.cell, self.cell, self.cell)

    def draw(self, world: World):
        p = self.p
        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        # grid
        for x in range(p.grid_width):
            for y in range(p.grid_height):
                rl.draw_rectangle_lines(int(x * self.cell), int(y * self.cell),
                                        int(self.cell), int(self.cell), rl.LIGHTGRAY)

        # bases & flags
        bx0, by0 = p.team0_base; bx1, by1 = p.team1_base
        fx0, fy0 = p.team0_flag; fx1, fy1 = p.team1_flag
        rl.draw_rectangle_rec(self._cell_rect(bx0, by0), rl.Color(220, 240, 255, 255))
        rl.draw_rectangle_rec(self._cell_rect(bx1, by1), rl.Color(255, 230, 220, 255))
        rl.draw_rectangle_rec(self._cell_rect(fx0, fy0), rl.Color(120, 160, 255, 120))
        rl.draw_rectangle_rec(self._cell_rect(fx1, fy1), rl.Color(255, 150, 120, 120))

        # agents
        pos = world.state.component_stores["GridPosition"].data
        team = world.state.component_stores["Team"].data[:, 0]
        carry = world.state.component_stores["CarryingOpponentFlag"].data[:, 0]

        for i in range(pos.shape[0]):
            x, y = int(pos[i, 0]), int(pos[i, 1])
            cx = int(x * self.cell + self.cell // 2)
            cy = int(y * self.cell + self.cell // 2)
            col = rl.BLUE if int(team[i]) == 0 else rl.RED
            rl.draw_circle(cx, cy, self.cell * 0.35, col)
            if bool(carry[i]):
                rl.draw_circle_lines(cx, cy, self.cell * 0.42, rl.GOLD)

        rl.end_drawing()

