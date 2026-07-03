# Multi-Agent Racing System Architecture Plan

## Overview
This document outlines the architecture and training pipeline for a next-generation racing AI system where every opponent is a unique, skilled agent with distinct personality traits, similar to *Need for Speed: Most Wanted*.

## 1. Core Driving Mechanics: Deep Reinforcement Learning (DRL)
The foundation of each agent's driving capability is built using DRL algorithms such as **Proximal Policy Optimization (PPO)** or **Soft Actor-Critic (SAC)**.

### State & Action Spaces
*   **Observation Space (Senses):** Raycasts (virtual lidar for track boundaries and obstacles), current velocity, steering angle, and distance/angle to the next track waypoints.
*   **Action Space (Muscles):** Continuous values for Steering (-1 to 1), Throttle (0 to 1), and Brake (0 to 1).
*   **Base Reward:** Positive reinforcement for forward progress along the track centerline; severe penalties for collisions or driving out of bounds.

## 2. Personality via Reward Shaping
Distinct driving styles are not hard-coded but defined by altering the reward functions during training.

*   **The Aggressor:** Receives massive bonus rewards for high-speed collisions with other cars. Penalties for vehicle damage are reduced or removed, teaching the agent to prioritize ramming over optimal racing lines.
*   **The Clean Racer:** Heavily penalized for sliding or physical contact. Rewarded for maintaining high average speeds and adhering strictly to the optimal racing line.
*   **The Show-off (Drifter):** Earns bonus points for maintaining high slip-angles (drifting) through corners, prioritizing style over pure speed.

## 3. Skill Level Distribution
To create varied difficulty levels without the AI feeling artificially "stupid" or erratic:

*   **Training Checkpoints:** Capture the neural network at different stages of its evolution.
    *   *Rookie:* Checkpoint at 500k steps (knows the track, but makes sub-optimal decisions).
    *   *Pro:* Checkpoint at 5M steps (near-perfect driving).
*   **Sensory Handicaps (Latency):** A "Boss" agent reacts every frame (e.g., 60Hz). A lower-skill agent's action updates are throttled (e.g., every 0.25 seconds), mimicking slower human reaction times.

## 4. Multi-Agent Dynamics: Self-Play & PBT
Agents must learn to race *against* each other, not just against the track.

*   **Self-Play:** Spawn multiple agents in the same environment. As they learn to drive, they simultaneously learn complex interactions like blocking, drafting, and overtaking.
*   **Population-Based Training (PBT):** Simulate evolution by training a large population. Top performers pass their neural weights to the next generation, while underperformers are discarded.

## 5. The "Brain Hierarchy": Combining DRL with LLMs
To achieve true, human-like strategy (grudges, dynamic rivalries), the system splits decision-making into two layers:

*   **Spinal Cord (Low-Level):** Specialized DRL models (`Drive_Fast`, `Ram_Target`, `Block_Pursuer`).
*   **Cortex (High-Level):** A fast LLM or State Machine evaluates the race context (e.g., "I am Razor, in 2nd place, and I hate the player in 1st"). It then hot-swaps the active DRL model on the fly (e.g., switching from `Drive_Fast` to `Ram_Target`).

## Implementation Phases
1.  **Engine Setup:** Initialize Unity ML-Agents or Unreal Engine learning environments.
2.  **Phase 1 (Basics):** Train base driving networks using PPO on empty tracks.
3.  **Phase 2 (Interaction):** Introduce Multi-Agent Self-Play, tweaking reward functions to breed different personality classes.
4.  **Phase 3 (Strategy):** Implement the high-level strategic brain to orchestrate the DRL models dynamically during a race.
