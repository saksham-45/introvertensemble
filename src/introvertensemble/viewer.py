from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
from PIL import Image, ImageDraw, ImageFont

from .simulation import LibrarySimulation, StepSummary
from .world import LibraryWorld


ZONE_COLORS = {
    "quiet_study": (120, 176, 126),
    "general_study": (220, 219, 154),
    "lounge_casual": (228, 181, 122),
    "collaboration": (129, 165, 217),
    "high_disturbance": (221, 122, 122),
    "staff_only": (165, 169, 174),
}

SEAT_COLORS = {
    "quiet_carrel": (74, 96, 123),
    "silent_carrel": (67, 87, 111),
    "deep_carrel": (56, 72, 91),
    "shared_table": (202, 187, 117),
    "individual_desk": (196, 155, 116),
    "lounge_chair": (181, 135, 103),
    "high_desk": (173, 167, 92),
    "booth": (167, 126, 109),
    "conference_chair": (162, 137, 137),
}

FEATURE_GRADIENTS = {
    "privacy": ((206, 231, 204), (38, 96, 58)),
    "static_noise": ((236, 243, 251), (192, 80, 77)),
    "foot_traffic": ((239, 243, 225), (214, 112, 57)),
    "future_crowding_risk": ((224, 237, 206), (215, 99, 99)),
    "stability": ((230, 240, 221), (57, 110, 72)),
    "wifi_strength": ((231, 243, 253), (54, 111, 186)),
    "interruption_risk": ((243, 239, 224), (179, 77, 56)),
}

AGENT_PROFILE_COLORS = {
    "focal_introvert": (237, 188, 58),
    "introvert": (91, 137, 196),
    "quiet_seeker": (93, 145, 98),
    "comfort_seeker": (184, 129, 78),
    "outlet_seeker": (149, 123, 189),
    "opportunistic_reseater": (204, 108, 96),
    "collaborator": (72, 150, 176),
    "standard": (148, 148, 148),
}


@dataclass(frozen=True)
class Viewport:
    width: int = 1400
    height: int = 920
    margin: int = 44
    sidebar_width: int = 260

    @property
    def world_width(self) -> int:
        return self.width - self.margin * 2 - self.sidebar_width

    @property
    def world_height(self) -> int:
        return self.height - self.margin * 2


class LibraryViewer:
    def __init__(
        self,
        world: LibraryWorld,
        viewport: Viewport | None = None,
        simulation: LibrarySimulation | None = None,
    ):
        self.world = world
        self.viewport = viewport or Viewport()
        self.spec = world.spec
        self.simulation = simulation
        self.feature_layer_name = "privacy"
        self.show_walk_graph = True
        self.show_labels = True
        self.show_seat_ids = False
        self.running_simulation = simulation is not None
        self.steps_per_second = 1.5
        self._step_accumulator_ms = 0.0
        self.last_summary: StepSummary | None = None
        self.clock = pygame.time.Clock()
        self._text_surface_cache: dict[tuple[str, int, bool, tuple[int, int, int]], pygame.Surface] = {}

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.viewport.width, self.viewport.height))
        pygame.display.set_caption("Introvert Ensemble Library Viewer")

        running = True
        while running:
            elapsed_ms = self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key, running)

            if self.simulation is not None and self.running_simulation:
                self._step_accumulator_ms += elapsed_ms
                ms_per_step = 1000.0 / max(0.25, self.steps_per_second)
                while self._step_accumulator_ms >= ms_per_step:
                    self.last_summary = self.simulation.step()
                    self._step_accumulator_ms -= ms_per_step

            self._draw(screen)
            pygame.display.flip()

        pygame.quit()

    def _handle_keydown(self, key: int, running: bool) -> bool:
        layer_names = list(self.spec.feature_layers)
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_g:
            self.show_walk_graph = not self.show_walk_graph
        elif key == pygame.K_l:
            self.show_labels = not self.show_labels
        elif key == pygame.K_i:
            self.show_seat_ids = not self.show_seat_ids
        elif key == pygame.K_RIGHT:
            index = layer_names.index(self.feature_layer_name)
            self.feature_layer_name = layer_names[(index + 1) % len(layer_names)]
        elif key == pygame.K_LEFT:
            index = layer_names.index(self.feature_layer_name)
            self.feature_layer_name = layer_names[(index - 1) % len(layer_names)]
        elif key == pygame.K_SPACE and self.simulation is not None:
            self.running_simulation = not self.running_simulation
        elif key == pygame.K_n and self.simulation is not None:
            self.last_summary = self.simulation.step()
        elif key == pygame.K_UP and self.simulation is not None:
            self.steps_per_second = min(8.0, self.steps_per_second + 0.5)
        elif key == pygame.K_DOWN and self.simulation is not None:
            self.steps_per_second = max(0.5, self.steps_per_second - 0.5)
        return running

    def _draw(self, screen: pygame.Surface) -> None:
        screen.fill((247, 247, 244))
        world_rect = pygame.Rect(
            self.viewport.margin,
            self.viewport.margin,
            self.viewport.world_width,
            self.viewport.world_height,
        )
        sidebar_rect = pygame.Rect(
            self.viewport.margin + self.viewport.world_width + 24,
            self.viewport.margin,
            self.viewport.sidebar_width - 24,
            self.viewport.world_height,
        )

        self._draw_world_frame(screen, world_rect)
        self._draw_feature_layer(screen, world_rect)
        self._draw_zones(screen, world_rect)
        self._draw_obstacles(screen, world_rect)
        if self.show_walk_graph:
            self._draw_walk_graph(screen, world_rect)
        self._draw_entrances(screen, world_rect)
        self._draw_event_overlays(screen, world_rect)
        self._draw_seats(screen, world_rect)
        self._draw_sidebar(screen, sidebar_rect)

    def _draw_world_frame(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, (241, 240, 235), rect, border_radius=10)
        pygame.draw.rect(screen, (72, 72, 72), rect, 2)
        for x in range(1, math.floor(self.spec.bounds.width)):
            start = self._to_screen(x, 0.0, rect)
            end = self._to_screen(x, self.spec.bounds.height, rect)
            pygame.draw.line(screen, (228, 228, 228), start, end, 1)
        for y in range(1, math.floor(self.spec.bounds.height)):
            start = self._to_screen(0.0, y, rect)
            end = self._to_screen(self.spec.bounds.width, y, rect)
            pygame.draw.line(screen, (228, 228, 228), start, end, 1)

    def _draw_feature_layer(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        layer_name = self.feature_layer_name
        for seat in self.spec.seats.values():
            value = self.world.feature_value_for_seat(seat.id, layer_name)
            color = _interpolate_color(*FEATURE_GRADIENTS[layer_name], value)
            px, py = self._to_screen(seat.x, seat.y, rect)
            radius = 28 if seat.seat_type.endswith("carrel") else 22
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (*color, 62), (radius, radius), radius)
            screen.blit(surface, (px - radius, py - radius))

    def _draw_zones(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        for zone in self.spec.zones.values():
            x1, y2 = self._to_screen(zone.x1, zone.y2, rect)
            x2, y1 = self._to_screen(zone.x2, zone.y1, rect)
            zone_rect = pygame.Rect(x1, y2, x2 - x1, y1 - y2)
            fill = ZONE_COLORS.get(zone.zone_type, (185, 185, 185))
            overlay = pygame.Surface((zone_rect.width, zone_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (*fill, 52), overlay.get_rect(), border_radius=8)
            screen.blit(overlay, zone_rect.topleft)
            pygame.draw.rect(screen, (95, 95, 95), zone_rect, 2, border_radius=8)
            if self.show_labels:
                self._draw_text(screen, zone.name, zone_rect.x + 6, zone_rect.y + 6, 16, True, (34, 34, 34))

    def _draw_obstacles(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        for obstacle in self.spec.obstacles:
            x1, y2 = self._to_screen(obstacle.x1, obstacle.y2, rect)
            x2, y1 = self._to_screen(obstacle.x2, obstacle.y1, rect)
            obs_rect = pygame.Rect(x1, y2, x2 - x1, y1 - y2)
            pygame.draw.rect(screen, (118, 123, 129), obs_rect, border_radius=6)
            pygame.draw.rect(screen, (84, 89, 95), obs_rect, 2, border_radius=6)

    def _draw_walk_graph(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        for edge in self.spec.graph_edges:
            start = self.spec.graph_nodes[edge.source]
            end = self.spec.graph_nodes[edge.target]
            pygame.draw.aaline(
                screen,
                (103, 103, 103),
                self._to_screen(start.x, start.y, rect),
                self._to_screen(end.x, end.y, rect),
            )
        for node in self.spec.graph_nodes.values():
            px, py = self._to_screen(node.x, node.y, rect)
            pygame.draw.circle(screen, (22, 22, 22), (px, py), 4)

    def _draw_entrances(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        for entrance in self.spec.entrances.values():
            px, py = self._to_screen(entrance.x, entrance.y, rect)
            pygame.draw.circle(screen, (30, 124, 196), (px, py), 10)
            pygame.draw.circle(screen, (255, 255, 255), (px, py), 10, 2)
            if self.show_labels:
                self._draw_text(screen, entrance.id, px - 10, py - 24, 14, True, (25, 25, 25))

    def _draw_event_overlays(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        if self.simulation is None or self.simulation.event_engine is None:
            return
        for event in self.simulation.event_engine.active_events:
            zone = self.spec.zones[event.zone_id]
            x1, y2 = self._to_screen(zone.x1, zone.y2, rect)
            x2, y1 = self._to_screen(zone.x2, zone.y1, rect)
            zone_rect = pygame.Rect(x1, y2, x2 - x1, y1 - y2)
            overlay = pygame.Surface((zone_rect.width, zone_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (210, 76, 71, 40), overlay.get_rect(), border_radius=8)
            screen.blit(overlay, zone_rect.topleft)
            pygame.draw.rect(screen, (163, 52, 46), zone_rect, 3, border_radius=8)

    def _draw_seats(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        for seat in self.spec.seats.values():
            px, py = self._to_screen(seat.x, seat.y, rect)
            occupant_id = self.world.occupancy[seat.id]
            occupied = occupant_id is not None
            color = SEAT_COLORS.get(seat.seat_type, (87, 87, 87))
            is_focal = occupant_id == getattr(self.simulation, "focal_agent_id", None)
            if occupied:
                color = self._seat_occupant_color(occupant_id)
            if is_focal:
                halo = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(halo, (247, 212, 74, 80), (20, 20), 18)
                screen.blit(halo, (px - 20, py - 20))
            self._draw_seat_glyph(screen, seat.seat_type, px, py, color)
            if occupied:
                outline_color = (248, 232, 143) if is_focal else (40, 40, 40)
                outline_radius = 15 if is_focal else 11
                outline_width = 3 if is_focal else 2
                pygame.draw.circle(screen, outline_color, (px, py), outline_radius, outline_width)
            if is_focal:
                self._draw_text(screen, "YOU", px + 10, py - 18, 12, True, (120, 78, 8))
            if self.show_labels and not self.show_seat_ids:
                short = seat.id.split("-")[0]
                self._draw_text(screen, short, px + 6, py - 6, 11, False, (40, 40, 40))
            if self.show_seat_ids:
                self._draw_text(screen, seat.id, px + 8, py - 4, 10, False, (40, 40, 40))

    def _draw_sidebar(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(screen, (254, 254, 252), rect)
        pygame.draw.rect(screen, (72, 72, 72), rect, 2, border_radius=8)
        y = rect.y + 16
        self._draw_text(screen, "Library Viewer", rect.x + 14, y, 22, True, (24, 24, 24))
        y += 38
        lines = [
            f"Layout: {self.spec.layout_id}",
            f"Feature layer: {self.feature_layer_name}",
            f"Seats: {len(self.spec.seats)}",
            f"Zones: {len(self.spec.zones)}",
            f"Entrances: {len(self.spec.entrances)}",
            f"Occupied: {sum(v is not None for v in self.world.occupancy.values())}",
        ]
        if self.simulation is not None:
            lines.extend(
                [
                    f"Step: {self.simulation.step_index}",
                    f"Hour: {self.simulation.current_hour:.2f}",
                    f"Playback: {'running' if self.running_simulation else 'paused'}",
                    f"Speed: {self.steps_per_second:.1f} step/s",
                ]
            )
        for line in lines:
            self._draw_text(screen, line, rect.x + 14, y, 15, False, (40, 40, 40))
            y += 24

        if self.last_summary is not None:
            y += 8
            self._draw_text(screen, "Latest Step", rect.x + 14, y, 15, False, (24, 24, 24))
            y += 24
            for line in (
                f"Arrivals: {self.last_summary.arrivals}",
                f"Departures: {self.last_summary.departures}",
                f"Reseats: {self.last_summary.reseats}",
            ):
                self._draw_text(screen, line, rect.x + 14, y, 14, False, (48, 48, 48))
                y += 22

        if self.simulation is not None and self.simulation.event_engine is not None:
            y += 8
            self._draw_text(screen, "Active Events", rect.x + 14, y, 15, False, (24, 24, 24))
            y += 24
            active_events = self.simulation.event_engine.active_events[:5]
            if not active_events:
                self._draw_text(screen, "None", rect.x + 14, y, 14, False, (72, 72, 72))
                y += 22
            for event in active_events:
                zone_name = self.spec.zones[event.zone_id].name
                self._draw_text(screen, f"{event.name} @ {zone_name}", rect.x + 14, y, 13, False, (72, 48, 48))
                y += 20

        focal = None
        if self.simulation is not None and self.simulation.focal_agent_id is not None:
            focal = self.simulation.agents.get(self.simulation.focal_agent_id)
        if focal is not None:
            y += 8
            self._draw_text(screen, "Focal Agent", rect.x + 14, y, 15, False, (24, 24, 24))
            y += 24
            focal_lines = [
                f"Seat: {focal.current_seat_id or 'None'}",
                f"Moves: {focal.total_moves}",
                f"Cooldown: {focal.cooldown_steps_remaining}",
                f"Preferred seat: {focal.dominant_seat_id() or 'None'}",
                f"Preferred zone: {focal.dominant_zone_id() or 'None'}",
            ]
            for line in focal_lines:
                self._draw_text(screen, line, rect.x + 14, y, 14, False, (48, 48, 48))
                y += 20
        elif self.simulation is not None:
            y += 8
            self._draw_text(screen, "Focal Agent", rect.x + 14, y, 15, False, (24, 24, 24))
            y += 24
            self._draw_text(screen, "Not currently in library", rect.x + 14, y, 14, False, (120, 54, 54))
            y += 20

        y += 8
        self._draw_text(screen, "Controls", rect.x + 14, y, 15, False, (24, 24, 24))
        y += 24
        controls = [
            "Left/Right: cycle feature layer",
            "G: toggle walk graph",
            "L: toggle labels",
            "I: toggle full seat ids",
            "Space: play/pause sim",
            "N: single simulation step",
            "Up/Down: sim speed",
            "Esc: quit",
        ]
        for line in controls:
            self._draw_text(screen, line, rect.x + 14, y, 14, False, (56, 56, 56))
            y += 22

        y += 10
        self._draw_text(screen, "Legend", rect.x + 14, y, 15, False, (24, 24, 24))
        y += 24
        for zone_type, color in ZONE_COLORS.items():
            pygame.draw.rect(screen, color, pygame.Rect(rect.x + 14, y + 4, 18, 12))
            pygame.draw.rect(screen, (31, 31, 31), pygame.Rect(rect.x + 14, y + 4, 18, 12), 1)
            self._draw_text(screen, zone_type, rect.x + 40, y, 15, False, (48, 48, 48))
            y += 22

        if self.simulation is not None:
            y += 10
            self._draw_text(screen, "Agent Colors", rect.x + 14, y, 15, False, (24, 24, 24))
            y += 24
            for profile_name in ("focal_introvert", "introvert", "quiet_seeker", "comfort_seeker", "outlet_seeker", "opportunistic_reseater", "collaborator", "standard"):
                color = AGENT_PROFILE_COLORS[profile_name]
                pygame.draw.circle(screen, color, (rect.x + 23, y + 10), 7)
                pygame.draw.circle(screen, (35, 35, 35), (rect.x + 23, y + 10), 7, 1)
                label = "focal_introvert (YOU)" if profile_name == "focal_introvert" else profile_name
                self._draw_text(screen, label, rect.x + 40, y, 14, False, (48, 48, 48))
                y += 20

    def _to_screen(self, x: float, y: float, rect: pygame.Rect) -> tuple[int, int]:
        sx = rect.x + int((x / self.spec.bounds.width) * rect.width)
        sy = rect.y + rect.height - int((y / self.spec.bounds.height) * rect.height)
        return sx, sy

    def _draw_seat_glyph(
        self,
        screen: pygame.Surface,
        seat_type: str,
        px: int,
        py: int,
        color: tuple[int, int, int],
    ) -> None:
        outline = (36, 36, 36)
        if seat_type in {"quiet_carrel", "silent_carrel", "deep_carrel"}:
            rect = pygame.Rect(px - 8, py - 7, 16, 14)
            pygame.draw.rect(screen, color, rect, border_radius=4)
            pygame.draw.rect(screen, outline, rect, 1, border_radius=4)
        elif seat_type == "individual_desk":
            rect = pygame.Rect(px - 7, py - 7, 14, 14)
            pygame.draw.rect(screen, color, rect, border_radius=3)
            pygame.draw.rect(screen, outline, rect, 1, border_radius=3)
        elif seat_type == "shared_table":
            rect = pygame.Rect(px - 9, py - 6, 18, 12)
            pygame.draw.ellipse(screen, color, rect)
            pygame.draw.ellipse(screen, outline, rect, 1)
        elif seat_type == "lounge_chair":
            pygame.draw.circle(screen, color, (px, py), 8)
            pygame.draw.circle(screen, outline, (px, py), 8, 1)
        elif seat_type == "high_desk":
            points = [(px, py - 8), (px + 8, py), (px, py + 8), (px - 8, py)]
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(screen, outline, points, 1)
        elif seat_type == "booth":
            rect = pygame.Rect(px - 10, py - 6, 20, 12)
            pygame.draw.rect(screen, color, rect, border_radius=5)
            pygame.draw.rect(screen, outline, rect, 1, border_radius=5)
        elif seat_type == "conference_chair":
            pygame.draw.circle(screen, color, (px, py), 6)
            pygame.draw.circle(screen, outline, (px, py), 6, 1)
        else:
            pygame.draw.circle(screen, color, (px, py), 7)
            pygame.draw.circle(screen, outline, (px, py), 7, 1)

    def _seat_occupant_color(self, occupant_id: str | None) -> tuple[int, int, int]:
        if occupant_id is None or self.simulation is None:
            return (182, 58, 58)
        agent = self.simulation.agents.get(occupant_id)
        if agent is None:
            return (182, 58, 58)
        return AGENT_PROFILE_COLORS.get(agent.profile.name, (182, 58, 58))

    def _draw_text(
        self,
        screen: pygame.Surface,
        text: str,
        x: int,
        y: int,
        size: int,
        bold: bool,
        color: tuple[int, int, int],
    ) -> None:
        key = (text, size, bold, color)
        surface = self._text_surface_cache.get(key)
        if surface is None:
            pil_font = ImageFont.load_default()
            bbox = pil_font.getbbox(text)
            width = max(1, bbox[2] - bbox[0] + 4)
            height = max(1, bbox[3] - bbox[1] + 4)
            image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(image)
            draw.text((2, 2), text, fill=(*color, 255), font=pil_font)
            mode = image.mode
            data = image.tobytes()
            surface = pygame.image.fromstring(data, image.size, mode).convert_alpha()
            self._text_surface_cache[key] = surface
        screen.blit(surface, (x, y))


def _interpolate_color(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(low, high))
