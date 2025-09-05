# SPDX-License-Identifier: MIT
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import jax.numpy as jnp

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
except Exception as e:
    Console = None  # soft dependency


@dataclass
class InspectorConfig:
    refresh_hz: float = 5.0  # UI refresh rate
    show_events: bool = True
    show_components: bool = True
    show_help: bool = True


@contextmanager
def _maybe_live(console):
    if console is None:
        # no-rich fallback: no-op context
        yield None
    else:
        with Live(refresh_per_second=30, console=console) as live:
            yield live


def _table_components(world) -> "Table":
    t = Table(title="Component Stores", show_header=True, header_style="bold")
    t.add_column("Name"); t.add_column("Shape"); t.add_column("Dtype"); t.add_column("Present / Capacity")
    for name, store in world.component_stores.items():
        shape = (world.capacity, *store.specification.shape)
        pres = int(jnp.sum(store.present_mask))
        t.add_row(name, str(shape), str(store.specification.dtype), f"{pres} / {world.capacity}")
    return t


def _table_events(world) -> "Table":
    t = Table(title="Event Buffers", show_header=True, header_style="bold")
    t.add_column("Name"); t.add_column("Payload Shape"); t.add_column("Dtype"); t.add_column("Count / Capacity")
    for name, buf in world.event_buffers.items():
        shape = (buf.specification.capacity, *buf.specification.shape)
        t.add_row(name, str(shape), str(buf.specification.dtype), f"{int(buf.count)} / {buf.specification.capacity}")
    return t


def _table_world(world) -> "Table":
    t = Table(title="World", show_header=False)
    alive = int(jnp.sum(world.alive_mask))
    t.add_row("Capacity", str(world.capacity))
    t.add_row("Alive", f"{alive} / {world.capacity}")
    t.add_row("Time Step", str(int(world.time_step)))
    return t


def run_inspector(world_provider, config: Optional[InspectorConfig] = None):
    """
    Live inspector. `world_provider` is a callable returning the latest WorldState (or World).
    Example:
        from ecsx.inspection.inspector_server import run_inspector
        run_inspector(lambda: world.state)
    """
    if config is None:
        config = InspectorConfig()
    console = Console() if Console is not None else None

    def render(world):
        if hasattr(world, "state"):  # allow World or WorldState
            world = world.state
        panels = []

        panels.append(Panel(_table_world(world), title="World"))

        if config.show_components:
            panels.append(Panel(_table_components(world), title="Components"))
        if config.show_events:
            panels.append(Panel(_table_events(world), title="Events"))
        if config.show_help:
            help_t = Table(title="Help", show_header=False)
            help_t.add_row("Inspector updates every", f"{config.refresh_hz:.1f} Hz")
            panels.append(Panel(help_t, title="Help"))

        from rich.layout import Layout
        layout = Layout()
        layout.split_column(
            Layout(panels[0], size=7),
            Layout(panels[1] if len(panels) > 1 else None),
            Layout(panels[2] if len(panels) > 2 else None),
            Layout(panels[3] if len(panels) > 3 else None),
        )
        return layout

    refresh_dt = max(0.05, 1.0 / config.refresh_hz)
    with _maybe_live(console) as live:
        try:
            while True:
                w = world_provider()
                r = render(w)
                if live is not None:
                    live.update(r)
                else:
                    # fallback: print once and exit
                    print(r)
                    return
                time.sleep(refresh_dt)
        except KeyboardInterrupt:
            return

