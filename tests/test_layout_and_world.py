from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from introvertensemble import LibraryWorld, load_layout


class LayoutAndWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")
        cls.world = LibraryWorld(cls.spec)

    def test_layout_counts(self) -> None:
        self.assertEqual(len(self.spec.zones), 9)
        self.assertEqual(len(self.spec.seats), 74)
        self.assertEqual(len(self.spec.entrances), 4)
        self.assertEqual(len(self.spec.graph_nodes), 18)
        self.assertEqual(len(self.spec.graph_edges), 19)

    def test_feature_values_are_bounded(self) -> None:
        sample_ids = ["QR-DC-01", "OR-ST-03", "HT-BT-01", "DH-T1-02"]
        for seat_id in sample_ids:
            for layer_name in self.spec.feature_layers:
                value = self.world.feature_value_for_seat(seat_id, layer_name)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_neighbors_and_crowding_ratio(self) -> None:
        neighbors = self.world.seat_neighbors("QR-SC-05")
        self.assertGreater(len(neighbors), 0)
        self.assertEqual(self.world.local_crowding_ratio("QR-SC-05"), 0.0)
        self.world.occupy_seat(neighbors[0].id, "agent_x")
        self.assertGreater(self.world.local_crowding_ratio("QR-SC-05"), 0.0)

    def test_path_costs_are_positive(self) -> None:
        self.assertGreater(self.world.path_cost_from_entrance_to_seat("E1", "QR-DC-01"), 0.0)
        self.assertGreater(self.world.path_cost_from_entrance_to_seat("F1", "RR-ID-03"), 0.0)

    def test_spawn_window_handles_boundary_wrap(self) -> None:
        self.assertEqual(self.world.active_spawn_window(21.9).id, "evening_quiet")
        self.assertEqual(self.world.active_spawn_window(22.0).id, "evening_quiet")
        self.assertEqual(self.world.active_spawn_window(22.2).id, "morning_open")
        self.assertEqual(self.world.active_spawn_window(7.0).id, "morning_open")


if __name__ == "__main__":
    unittest.main()
