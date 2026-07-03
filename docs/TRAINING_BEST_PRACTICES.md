# Training best practices → concrete decisions for introvertensemble

Synthesized from primary sources (July 2026). Each finding maps to a specific
choice for our env: a single focal agent, 11 discrete actions, ~141-dim Box obs,
dense per-step reward, short (~24-step) episodes, CPU-trainable, PPO/SB3.

---

## 1. PPO hyperparameters (highest-leverage knobs first)

From Andrychowicz et al., *What Matters in On-Policy RL?* (2020) — a 250k-run
empirical study — and the SB3 tips / RL Zoo.

| Knob | Evidence-based recommendation | Our choice |
|------|-------------------------------|------------|
| **Observation normalization** | "**Always** use observation normalization"; SB3: "always normalize the input… using `VecNormalize`". Single highest-leverage preprocessing step. | `VecNormalize(norm_obs=True)`, stats saved with model, `training=False` at eval |
| Policy loss / clip | PPO loss; start clip **0.25**, try lower/higher | `clip_range=0.2` (SB3 default; within range) |
| GAE λ | **0.9** (paper); 0.95 common default | `gae_lambda=0.95` |
| Discount γ | "one of the most important"; start **0.99**, tune per env | `gamma=0.99` (episode ≈24 steps ≪ horizon 100, safe) |
| Optimizer / LR | Adam, β₁=0.9, LR **3e-4** safe default | `learning_rate=3e-4` |
| Activation / arch | **tanh**; 2 hidden layers; **wide value MLP not shared with policy** | `net_arch=dict(pi=[256,256], vf=[256,256])`, `activation_fn=Tanh` |
| Last policy layer init | "initialize with 100× smaller weights" | SB3 `ortho_init=True` already scales the policy head by gain 0.01 ✓ |
| Epochs / minibatch | multiple passes; shuffle transitions; recompute advantages per pass | `n_epochs=10`, `batch_size=256` |
| Rollout length | tune #transitions per iteration | `n_steps=2048` (single env) |
| Entropy | regularizers rarely help much, but entropy aids discrete exploration | `ent_coef=0.01` (was 0.02 — lowered) |
| Value-fn normalization | "check if it improves" | `VecNormalize(norm_reward=True, clip_reward=10)`, verify vs off |
| Grad clip | "slightly helps, secondary" | `max_grad_norm=0.5` (SB3 default) |

**Model size matters for generalization** (ProcGen): larger nets improve both
sample efficiency and generalization → we go [256,256], not [64,64].

---

## 2. Procedural generation & domain randomization

From Cobbe et al., *Leveraging Procedural Generation to Benchmark RL* (ProcGen,
ICML 2020) and Tobin et al., *Domain Randomization* (2017).

- **Agents overfit even to large training sets.** "Agents exhibit the capacity
  to overfit to remarkably large training sets" → a fixed handful of layouts is
  memorized, not learned. **4 hand-made layouts cannot support a generalization
  claim.**
- **Train on many, test on the full unseen distribution.** ProcGen's protocol:
  train on a fixed finite set (e.g. **200 levels**), evaluate on the *full
  distribution*. The train-vs-test gap **is** the generalization measure.
- **More training levels → smaller gap.** Diverse environment distributions are
  "essential to adequately train and evaluate RL agents."

**Our protocol:**
- Generate **≥64 training layouts** (aim 128–200), **16 validation**, **16 test**
  layouts the agent never trains on, from disjoint generator seed ranges.
- Randomize **per episode**: layout (from train pool), population/profile mix,
  arrival rates, session length, crowd pressure.
- Report **train-pool return vs held-out-test return**; the gap is the headline
  generalization number, not a single in-distribution score.

---

## 3. Reward design (avoid specification gaming)

From Ng, Harada & Russell (1999), *Policy Invariance under Reward
Transformations* (potential-based reward shaping, PBRS).

- **PBRS is the only shaping guaranteed not to change the optimal policy** —
  shaping of the form `F(s,s') = γ·Φ(s') − Φ(s)`. "Arbitrary changes to the
  reward function may result in reward hacking."
- Our corrected reward is already hardened by construction: relocation is
  **charged** (a real cost, not shaping) and farmable habit terms are **excluded**
  from the RL reward. We keep the objective honest rather than bolt on shaping.
- **If** we later want faster learning, the principled route is PBRS with a
  potential like `Φ(s) = best available environment score` — not an ad-hoc bonus.
  Deferred; not needed for correctness.

---

## 4. Rigorous evaluation

From Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*
(NeurIPS 2021, Outstanding Paper) + the `rliable` library.

- **Report IQM (interquartile mean) with stratified-bootstrap 95% CIs**, plus
  performance profiles — not mean ± std, which is dominated by outliers.
- **Percentile CIs are reliable from as few as N=10 runs.** Use **≥10 training
  seeds** where compute allows, **≥5** minimum; the unit of analysis is the
  *training run*, not the eval episode.
- **Eval episodes:** SB3 recommends 5–20 per config with `deterministic=True`;
  we use ≥30 paired episodes, which is comfortably sufficient.
- Report held-out generalization honestly (§2). Our `make_results_table.py`
  already refuses under-powered tables.

---

## 5. Observation & action design

- **Withhold the reward signal from observations** — otherwise the task is a
  disguised bandit. We already do this (`score_in_obs=False`). Present candidates
  in a **consistent canonical order** (we sort by the internal score) so the MLP
  sees a stable slot structure.
- **Action masking beats penalizing invalid actions.** MaskablePPO (sb3-contrib)
  replaces invalid-action logits with −∞: "higher training efficiency and scales
  better with an increasing number of actions than penalizing." For us the
  invalid actions are **padded candidate slots** (when <5 nearby/global seats
  exist) and **stay-only under cooldown**. Worth adding as an enhancement arm;
  gains are modest here because our action space is small (11) and candidate
  lists already exclude occupied seats.

---

## 6. Curriculum learning

From the curriculum-RL literature (CURATE 2024; surveys).

- Curricula "improve learning efficiency and help tackle more challenging tasks,"
  but the strongest evidence is for **sparse-reward / hard-exploration** tasks.
- Our reward is **dense** (per-step seat quality), so curriculum is **optional**,
  not essential. We expose a simple crowd-density ramp (low → high arrival rate)
  as an ablation, default off, and only keep it if it measurably helps.

---

## Resulting training recipe (what we will run)

1. Generate 128 train / 16 val / 16 test procedural layouts (disjoint seeds).
2. `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10)`
   around the env; **save the running stats with the model**.
3. PPO(MlpPolicy, tanh, pi/vf=[256,256], lr=3e-4, n_steps=2048, batch=256,
   n_epochs=10, γ=0.99, λ=0.95, clip=0.2, ent=0.01), environment reward, no
   score leak, random spawn, domain randomization on.
4. Train ≥3 seeds (≥1M steps each; more if it keeps improving); pick best by
   **validation** return, never test.
5. Evaluate the frozen best model on the **16 held-out test layouts** vs Random /
   No-op / Greedy / best-seat, ≥30 paired episodes, IQM + bootstrap CIs; report
   the train-vs-test generalization gap.

## Sources
- Andrychowicz et al. 2020, *What Matters In On-Policy RL?* — https://arxiv.org/abs/2006.05990
- Cobbe et al. 2020, *Leveraging Procedural Generation to Benchmark RL (ProcGen)* — https://arxiv.org/abs/1912.01588
- Tobin et al. 2017, *Domain Randomization* — https://arxiv.org/abs/1703.06907
- Ng, Harada & Russell 1999, *Policy Invariance under Reward Transformations* — https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice* — https://arxiv.org/abs/2108.13264 · rliable: https://github.com/google-research/rliable
- Stable-Baselines3 RL Tips & Tricks — https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- RL Baselines3 Zoo (tuned hyperparameters) — https://github.com/DLR-RM/rl-baselines3-zoo
- sb3-contrib MaskablePPO — https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
