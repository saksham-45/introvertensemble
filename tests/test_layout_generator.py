"""Tests for the procedural layout generator."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from introvertensemble.layout_generator import generate_layout_dict, write_layout
from introvertensemble.loader import load_layout
from introvertensemble.simulation import LibrarySimulation, SimulationConfig
from introvertensemble.world import LibraryWorld


class TestLayoutGenerator(unittest.TestCase):
    def test_generated_layout_loads_and_validates(self) -> None:
        """Every generated layout must pass load_layout's validator."""
        import tempfile
        for seed in (1000, 1234, 4242):
            with tempfile.TemporaryDirectory() as tmp:
                layout_dir = write_layout(Path(tmp) / f"gen_{seed}", seed=seed)
                spec = load_layout(layout_dir)
                self.assertGreater(len(spec.seats), 10)
                self.assertGreater(len(spec.zones), 3)
                # World construction precomputes all-pairs shortest paths, which
                # only succeeds if the walk graph is connected.
                LibraryWorld(spec)

    def test_generated_layout_simulates(self) -> None:
        """A generated layout must run a full simulation without error."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            layout_dir = write_layout(Path(tmp) / "gen", seed=777, difficulty=0.8)
            world = LibraryWorld(load_layout(layout_dir))
            sim = LibrarySimulation(world, config=SimulationConfig(events_enabled=True), seed=0)
            for _ in range(40):
                sim.step()

    def test_generation_is_deterministic(self) -> None:
        """Same seed => identical layout (reproducibility)."""
        self.assertEqual(generate_layout_dict(2024), generate_layout_dict(2024))
        self.assertNotEqual(generate_layout_dict(2024), generate_layout_dict(2025))

    def test_canonical_event_zones_present(self) -> None:
        """Canonical zone ids must appear so default events can fire."""
        bundle = generate_layout_dict(99)
        zone_ids = {z["id"] for z in bundle["zones.json"]}
        # At least the quiet + collaboration canonical ids should be emitted.
        self.assertIn("z_quiet_room_1", zone_ids)
        self.assertIn("z_conference_room_1", zone_ids)

    def test_every_feature_layer_covers_every_zone(self) -> None:
        """The scorer indexes feature layers by zone id; all must be present."""
        bundle = generate_layout_dict(55)
        zone_ids = {z["id"] for z in bundle["zones.json"]}
        for layer in bundle["feature_fields.json"]["layers"]:
            self.assertEqual(set(layer["zone_defaults"].keys()), zone_ids)


if __name__ == "__main__":
    unittest.main()
