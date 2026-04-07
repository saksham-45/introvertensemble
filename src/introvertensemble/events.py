from __future__ import annotations

import random
from dataclasses import dataclass

from .models import ActiveEvent, EventTemplate


DEFAULT_EVENT_TEMPLATES: tuple[EventTemplate, ...] = (
    EventTemplate(
        id="collaboration_burst",
        name="Collaboration Burst",
        zone_ids=("z_discussion_room_1", "z_conference_room_1"),
        probability_per_step=0.12,
        min_duration_steps=2,
        max_duration_steps=5,
        layer_deltas={"static_noise": 0.16, "interruption_risk": 0.12, "privacy": -0.10},
    ),
    EventTemplate(
        id="printer_queue",
        name="Printer Queue",
        zone_ids=("z_high_turnover_1",),
        probability_per_step=0.16,
        min_duration_steps=2,
        max_duration_steps=4,
        layer_deltas={"foot_traffic": 0.18, "interruption_risk": 0.22, "privacy": -0.08},
    ),
    EventTemplate(
        id="cafe_rush",
        name="Cafe Rush",
        zone_ids=("z_cafe_pantry_1",),
        probability_per_step=0.14,
        min_duration_steps=2,
        max_duration_steps=4,
        layer_deltas={"static_noise": 0.20, "foot_traffic": 0.16, "interruption_risk": 0.14},
    ),
    EventTemplate(
        id="quiet_room_disturbance",
        name="Quiet Room Disturbance",
        zone_ids=("z_quiet_room_1",),
        probability_per_step=0.05,
        min_duration_steps=1,
        max_duration_steps=3,
        layer_deltas={"static_noise": 0.08, "interruption_risk": 0.10, "stability": -0.10},
    ),
)


@dataclass
class EventStepSummary:
    activated: list[ActiveEvent]
    expired: list[str]
    active_events: list[ActiveEvent]


class EventEngine:
    def __init__(
        self,
        templates: tuple[EventTemplate, ...] | None = None,
        seed: int = 7,
    ):
        self.templates = templates or DEFAULT_EVENT_TEMPLATES
        self.random = random.Random(seed)
        self.active_events: list[ActiveEvent] = []
        self._next_event_index = 1

    def step(self) -> EventStepSummary:
        expired: list[str] = []
        still_active: list[ActiveEvent] = []
        for event in self.active_events:
            event.remaining_steps -= 1
            if event.remaining_steps <= 0:
                expired.append(event.id)
            else:
                still_active.append(event)
        self.active_events = still_active

        activated: list[ActiveEvent] = []
        for template in self.templates:
            active_count = sum(1 for event in self.active_events if event.template_id == template.id)
            if active_count >= template.max_active_instances:
                continue
            if self.random.random() > template.probability_per_step:
                continue
            zone_id = self.random.choice(template.zone_ids)
            duration = self.random.randint(template.min_duration_steps, template.max_duration_steps)
            event = ActiveEvent(
                id=f"event_{self._next_event_index:04d}",
                template_id=template.id,
                name=template.name,
                zone_id=zone_id,
                remaining_steps=duration,
                layer_deltas=dict(template.layer_deltas),
            )
            self._next_event_index += 1
            activated.append(event)
            self.active_events.append(event)
        return EventStepSummary(
            activated=activated,
            expired=expired,
            active_events=list(self.active_events),
        )

    def zone_layer_deltas(self) -> dict[str, dict[str, float]]:
        deltas: dict[str, dict[str, float]] = {}
        for event in self.active_events:
            zone_deltas = deltas.setdefault(event.zone_id, {})
            for layer_name, delta in event.layer_deltas.items():
                zone_deltas[layer_name] = zone_deltas.get(layer_name, 0.0) + delta
        return deltas
