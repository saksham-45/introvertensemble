from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from introvertensemble import LibrarySimulation, LibraryWorld, ObservationBuilder, load_layout
from introvertensemble.metrics import run_episode
from introvertensemble.events import EventEngine, EventTemplate
from introvertensemble.simulation import SimulationConfig


class EventsAndObservationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")

    def test_event_engine_activates_and_aggregates_zone_deltas(self) -> None:
        engine = EventEngine(
            templates=(
                EventTemplate(
                    id="test_event",
                    name="Test Event",
                    zone_ids=("z_quiet_room_1",),
                    probability_per_step=1.0,
                    min_duration_steps=2,
                    max_duration_steps=2,
                    layer_deltas={"static_noise": 0.2, "privacy": -0.1},
                ),
            ),
            seed=3,
        )
        summary = engine.step()
        self.assertEqual(len(summary.activated), 1)
        deltas = engine.zone_layer_deltas()
        self.assertIn("z_quiet_room_1", deltas)
        self.assertAlmostEqual(deltas["z_quiet_room_1"]["static_noise"], 0.2)

    def test_world_feature_value_reflects_dynamic_event_deltas(self) -> None:
        world = LibraryWorld(self.spec)
        base_noise = world.feature_value_for_seat("QR-SC-05", "static_noise")
        world.set_dynamic_zone_layer_deltas({"z_quiet_room_1": {"static_noise": 0.15}})
        boosted_noise = world.feature_value_for_seat("QR-SC-05", "static_noise")
        self.assertGreater(boosted_noise, base_noise)

    def test_focal_observation_contains_candidate_lists(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=("QR-SC-05", "QR-SC-05", "QR-SC-04"),
                focal_agent_session_steps=12,
            ),
            seed=8,
        )
        builder = ObservationBuilder(simulation, top_k=3)
        observation = builder.build_focal_observation()
        self.assertEqual(observation.current_seat_id, simulation.agents["focal_agent"].current_seat_id)
        self.assertLessEqual(len(observation.nearby_candidates), 3)
        self.assertLessEqual(len(observation.global_candidates), 3)
        if observation.global_candidates:
            self.assertGreaterEqual(observation.global_candidates[0].score, observation.global_candidates[-1].score)

    def test_events_are_visible_in_focal_observation(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                events_enabled=False,
            ),
            seed=5,
        )
        simulation.event_engine = EventEngine(
            templates=(
                EventTemplate(
                    id="quiet_disturbance",
                    name="Quiet Disturbance",
                    zone_ids=("z_quiet_room_1",),
                    probability_per_step=1.0,
                    min_duration_steps=2,
                    max_duration_steps=2,
                    layer_deltas={"static_noise": 0.1},
                ),
            ),
            seed=1,
        )
        simulation._step_events()
        observation = ObservationBuilder(simulation).build_focal_observation()
        self.assertTrue(observation.active_events)
        self.assertEqual(observation.active_events[0][0], "z_quiet_room_1")

    def test_episode_metrics_counts_event_activations_not_active_duration(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                events_enabled=False,
            ),
            seed=5,
        )
        simulation.event_engine = EventEngine(
            templates=(
                EventTemplate(
                    id="long_event",
                    name="Long Event",
                    zone_ids=("z_quiet_room_1",),
                    probability_per_step=1.0,
                    min_duration_steps=3,
                    max_duration_steps=3,
                    max_active_instances=1,
                    layer_deltas={"static_noise": 0.1},
                ),
            ),
            seed=1,
        )
        metrics = run_episode(simulation, steps=3)
        self.assertEqual(metrics.total_event_activations, 1)


if __name__ == "__main__":
    unittest.main()
