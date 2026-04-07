from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from introvertensemble import LibrarySimulation, LibraryWorld, load_layout
from introvertensemble.agents import AgentProfile, SimAgent
from introvertensemble.metrics import run_episode
from introvertensemble.scoring import SeatScorer
from introvertensemble.simulation import SimulationConfig


class ScoringAndSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")

    def test_introvert_prefers_quiet_seat_over_discussion_seat(self) -> None:
        world = LibraryWorld(self.spec)
        scorer = SeatScorer(world)
        intro = SimAgent(id="intro", profile=AgentProfile.introvert(), entrance_id="E1", session_steps_remaining=8)
        quiet = scorer.score_seat(intro, "QR-DC-01", origin_entrance_id="E1").total
        discussion = scorer.score_seat(intro, "DH-T1-02", origin_entrance_id="E1").total
        self.assertGreater(quiet, discussion)

    def test_focal_current_seat_gets_dwell_bonus(self) -> None:
        world = LibraryWorld(self.spec)
        scorer = SeatScorer(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=8,
            role="focal",
        )
        world.occupy_seat("QR-SC-05", focal.id)
        focal.record_seat("QR-SC-05", self.spec.seats["QR-SC-05"].zone_id)

        early = scorer.score_seat(focal, "QR-SC-05").total
        focal.steps_in_current_seat = 6
        settled = scorer.score_seat(focal, "QR-SC-05").total
        self.assertGreater(settled, early)

    def test_focal_agent_prefers_exact_habit_seat_on_arrival(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=8,
            role="focal",
        )
        for _ in range(3):
            focal.record_seat("QR-SC-05", self.spec.seats["QR-SC-05"].zone_id)
            focal.current_seat_id = None
        chosen = simulation._choose_arrival_seat(focal)
        self.assertEqual(chosen, "QR-SC-05")

    def test_focal_agent_prefers_nearby_fallback_when_habit_seat_taken(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        world.occupy_seat("QR-SC-05", "other_agent")
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=8,
            role="focal",
            local_search_radius=2.0,
            arrival_acceptability_threshold=-1.0,
        )
        for _ in range(3):
            focal.record_seat("QR-SC-05", self.spec.seats["QR-SC-05"].zone_id)
            focal.current_seat_id = None

        chosen = simulation._choose_arrival_seat(focal)
        self.assertIsNotNone(chosen)
        self.assertEqual(self.spec.seats[chosen].zone_id, "z_quiet_room_1")
        preferred = self.spec.seats["QR-SC-05"]
        fallback = self.spec.seats[chosen]
        distance = ((preferred.x - fallback.x) ** 2 + (preferred.y - fallback.y) ** 2) ** 0.5
        self.assertLessEqual(distance, focal.local_search_radius)

    def test_standard_prefers_available_open_reading_when_quiet_room_is_full(self) -> None:
        world = LibraryWorld(self.spec)
        for seat_id, seat in self.spec.seats.items():
            if seat.zone_id == "z_quiet_room_1":
                world.occupy_seat(seat_id, f"fill_{seat_id}")
        scorer = SeatScorer(world)
        standard = SimAgent(id="std", profile=AgentProfile.standard(), entrance_id="E1", session_steps_remaining=8)
        best_seat = None
        best_score = float("-inf")
        for seat in world.available_seats():
            score = scorer.score_seat(standard, seat.id, origin_entrance_id="E1").total
            if score > best_score:
                best_score = score
                best_seat = seat.id
        self.assertIsNotNone(best_seat)
        self.assertEqual(self.spec.seats[best_seat].zone_id, "z_open_reading_1")

    def test_simulation_occupancy_matches_active_agents(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        for _ in range(80):
            simulation.step()
            occupied = sum(value is not None for value in world.occupancy.values())
            seated_agents = sum(1 for agent in simulation.agents.values() if agent.current_seat_id is not None)
            self.assertEqual(occupied, seated_agents)
            self.assertLessEqual(occupied, len(self.spec.seats))

    def test_simulation_wraps_after_end_of_schedule(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        for _ in range(80):
            simulation.step()
        self.assertGreaterEqual(simulation.current_hour, 8.0)
        self.assertLess(simulation.current_hour, 22.0)

    def test_focal_agent_stays_put_when_current_seat_is_good_enough(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=10,
            role="focal",
        )
        world.occupy_seat("QR-SC-05", focal.id)
        focal.record_seat("QR-SC-05", self.spec.seats["QR-SC-05"].zone_id)
        simulation.agents[focal.id] = focal

        moved = simulation._process_focal_reseat(focal)
        self.assertFalse(moved)
        self.assertEqual(focal.current_seat_id, "QR-SC-05")

    def test_focal_agent_moves_only_when_current_location_becomes_bad(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E4",
            session_steps_remaining=10,
            role="focal",
            leave_threshold=1.20,
            switching_improvement_threshold=0.10,
        )
        world.occupy_seat("DH-T1-02", focal.id)
        focal.record_seat("DH-T1-02", self.spec.seats["DH-T1-02"].zone_id)
        simulation.agents[focal.id] = focal

        for seat_id in ["DH-T1-01", "DH-T1-03", "DH-T1-04", "DH-T1-05", "DH-T1-06", "DH-T2-01", "DH-T2-03"]:
            world.occupy_seat(seat_id, f"neighbor_{seat_id}")
        for seat_id in self.spec.seats:
            if seat_id in {"DH-T1-02", "QR-DC-01", "QR-SC-05", "QR-SC-04"}:
                continue
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"block_{seat_id}")

        moved = simulation._process_focal_reseat(focal)
        self.assertTrue(moved)
        self.assertIn(focal.current_seat_id, {"QR-DC-01", "QR-SC-05", "QR-SC-04"})
        self.assertIsNone(world.occupancy["DH-T1-02"])

    def test_focal_agent_is_evaluated_each_step_not_sampled_by_background_budget(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=("DH-T1-02", "DH-T1-02", "DH-T1-02"),
                focal_agent_entrance_id="E4",
                focal_agent_session_steps=12,
            ),
            seed=13,
        )
        focal = simulation.agents["focal_agent"]
        focal.leave_threshold = 1.20
        focal.switching_improvement_threshold = 0.10
        for seat_id in ["DH-T1-01", "DH-T1-03", "DH-T1-04", "DH-T1-05", "DH-T1-06", "DH-T2-01", "DH-T2-03"]:
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"neighbor_{seat_id}")
        for seat_id in self.spec.seats:
            if seat_id in {"DH-T1-02", "QR-DC-01", "QR-SC-05", "QR-SC-04"}:
                continue
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"block_{seat_id}")

        simulation._sample_rate = lambda rate: 0  # type: ignore[method-assign]
        simulation._process_reseating(world.active_spawn_window(10.0))
        self.assertNotEqual(focal.current_seat_id, "DH-T1-02")
        self.assertIn(focal.current_seat_id, {"QR-DC-01", "QR-SC-05", "QR-SC-04"})

    def test_focal_move_cooldown_prevents_immediate_repeated_reseating(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=("DH-T1-02", "DH-T1-02", "DH-T1-02"),
                focal_agent_entrance_id="E4",
                focal_agent_session_steps=16,
                focal_move_cooldown_steps=3,
            ),
            seed=13,
        )
        focal = simulation.agents["focal_agent"]
        focal.leave_threshold = 1.20
        focal.switching_improvement_threshold = 0.10
        for seat_id in ["DH-T1-01", "DH-T1-03", "DH-T1-04", "DH-T1-05", "DH-T1-06", "DH-T2-01", "DH-T2-03"]:
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"neighbor_{seat_id}")
        for seat_id in self.spec.seats:
            if seat_id in {"DH-T1-02", "QR-DC-01", "QR-SC-05", "QR-SC-04"}:
                continue
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"block_{seat_id}")

        simulation._sample_rate = lambda rate: 0  # type: ignore[method-assign]
        simulation._process_reseating(world.active_spawn_window(10.0))
        moved_to = focal.current_seat_id
        first_moves = focal.total_moves
        simulation._process_reseating(world.active_spawn_window(10.25))
        self.assertEqual(focal.current_seat_id, moved_to)
        self.assertEqual(focal.total_moves, first_moves)

    def test_focal_rejects_crowded_fallback_even_if_it_is_available(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=10,
            role="focal",
            arrival_acceptability_threshold=-1.0,
        )
        crowded_fallback = "QR-SC-05"
        for neighbor in world.seat_neighbors(crowded_fallback):
            world.occupy_seat(neighbor.id, f"crowd_{neighbor.id}")

        self.assertFalse(simulation._is_focal_fallback_acceptable(focal, crowded_fallback))

    def test_focal_does_not_make_small_same_zone_micro_move(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=10,
            role="focal",
            stay_threshold=10.0,
            leave_threshold=-10.0,
            switching_improvement_threshold=0.10,
        )
        current_seat = "QR-SC-05"
        candidate_seat = "QR-SC-04"
        world.occupy_seat(current_seat, focal.id)
        focal.record_seat(current_seat, self.spec.seats[current_seat].zone_id)
        focal.steps_in_current_seat = 6
        simulation.agents[focal.id] = focal

        for seat_id in ("QR-SC-06", "QR-DC-03"):
            world.occupy_seat(seat_id, f"crowd_{seat_id}")

        moved = simulation._process_focal_reseat(focal)
        self.assertFalse(moved)
        self.assertEqual(focal.current_seat_id, current_seat)

    def test_focal_can_make_same_zone_move_when_rewarding_enough(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E1",
            session_steps_remaining=10,
            role="focal",
            stay_threshold=10.0,
            leave_threshold=3.20,
            switching_improvement_threshold=0.10,
            arrival_acceptability_threshold=-1.0,
        )
        current_seat = "QR-SC-05"
        world.occupy_seat(current_seat, focal.id)
        focal.record_seat(current_seat, self.spec.seats[current_seat].zone_id)
        focal.steps_in_current_seat = 6
        simulation.agents[focal.id] = focal

        for seat_id in ("QR-SC-01", "QR-SC-02", "QR-SC-03", "QR-SC-04", "QR-SC-06", "QR-DC-01", "QR-DC-02", "QR-DC-03"):
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"crowd_{seat_id}")

        moved = simulation._process_focal_reseat(focal)
        self.assertTrue(moved)
        self.assertNotEqual(focal.current_seat_id, current_seat)
        self.assertEqual(self.spec.seats[focal.current_seat_id].zone_id, "z_quiet_room_1")
        self.assertGreater(
            simulation.scorer.score_seat(focal, focal.current_seat_id).total,
            simulation.scorer.score_seat(focal, current_seat).total,
        )

    def test_focal_bad_seat_does_not_force_move_into_unacceptable_fallback(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(world)
        focal = SimAgent(
            id="focal",
            profile=AgentProfile.focal_introvert(),
            entrance_id="E4",
            session_steps_remaining=10,
            role="focal",
            leave_threshold=2.5,
            switching_improvement_threshold=0.1,
            arrival_acceptability_threshold=-1.0,
        )
        world.occupy_seat("DH-T1-02", focal.id)
        focal.record_seat("DH-T1-02", self.spec.seats["DH-T1-02"].zone_id)
        simulation.agents[focal.id] = focal

        open_fallback = "QR-SC-05"
        for neighbor in world.seat_neighbors(open_fallback):
            if world.occupancy[neighbor.id] is None:
                world.occupy_seat(neighbor.id, f"crowd_{neighbor.id}")

        for seat_id in self.spec.seats:
            if seat_id in {"DH-T1-02", open_fallback}:
                continue
            if world.occupancy[seat_id] is None:
                world.occupy_seat(seat_id, f"block_{seat_id}")

        moved = simulation._process_focal_reseat(focal)
        self.assertFalse(moved)
        self.assertEqual(focal.current_seat_id, "DH-T1-02")

    def test_forced_reseating_can_trigger(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(step_minutes=15, introvert_share=1.0, reseat_margin=0.0),
            seed=11,
        )
        agent = SimAgent(
            id="agent_0001",
            profile=AgentProfile.introvert(),
            entrance_id="E4",
            session_steps_remaining=10,
            current_seat_id="DH-T1-02",
        )
        world.occupy_seat("DH-T1-02", agent.id)
        agent.record_seat("DH-T1-02", self.spec.seats["DH-T1-02"].zone_id)
        simulation.agents[agent.id] = agent

        target_better = "QR-DC-01"
        for seat_id in self.spec.seats:
            if seat_id in {agent.current_seat_id, target_better}:
                continue
            world.occupy_seat(seat_id, f"block_{seat_id}")

        reseats = simulation._process_reseating(world.active_spawn_window(10.0))
        self.assertEqual(reseats, 1)
        self.assertEqual(agent.current_seat_id, target_better)
        self.assertEqual(world.occupancy[target_better], agent.id)
        self.assertIsNone(world.occupancy["DH-T1-02"])

    def test_collaborator_prefers_collaboration_over_quiet(self) -> None:
        world = LibraryWorld(self.spec)
        scorer = SeatScorer(world)
        collaborator = SimAgent(
            id="collab",
            profile=AgentProfile.collaborator(),
            entrance_id="E4",
            session_steps_remaining=8,
        )
        discussion = scorer.score_seat(collaborator, "DH-T1-02", origin_entrance_id="E4").total
        quiet = scorer.score_seat(collaborator, "QR-DC-01", origin_entrance_id="E4").total
        self.assertGreater(discussion, quiet)

    def test_background_profile_mix_can_force_single_archetype(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                introvert_share=0.0,
                background_profile_mix=(("collaborator", 1.0),),
            ),
            seed=5,
        )
        profiles = {simulation._create_agent().profile.name for _ in range(10)}
        self.assertEqual(profiles, {"collaborator"})

    def test_run_episode_emits_focal_metrics(self) -> None:
        world = LibraryWorld(self.spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=("QR-SC-05", "QR-SC-05", "QR-SC-04"),
                focal_agent_session_steps=20,
            ),
            seed=17,
        )
        metrics = run_episode(simulation, steps=16)
        self.assertEqual(metrics.steps, 16)
        self.assertTrue(metrics.focal_present)
        self.assertIsNotNone(metrics.focal_average_score)
        self.assertGreaterEqual(metrics.average_occupancy, 0.0)
        self.assertLessEqual(metrics.peak_occupancy, len(self.spec.seats))


if __name__ == "__main__":
    unittest.main()
