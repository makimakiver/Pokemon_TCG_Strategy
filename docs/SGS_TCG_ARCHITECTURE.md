# SGS for the Pokemon TCG (cabt) environment

An adaptation of **Self-Guided Self-Play (SGS)** (Bailey, Wen, Dong, Hashimoto, Ma — arXiv:2604.20209, 2026)
to the `cabt` "Limited Card Battle" engine in this repo.

SGS was built for formal theorem proving in Lean4. The reason it ports cleanly to Pokemon TCG is that
**this engine is already a verifier** (like the Lean4 compiler): a scenario is "proved" when the Solver
wins, and win/loss is ground truth from `battle_select`. The engine also ships a state-cloning search API
(`search_begin` / `search_step` / `search_release`) that lets a Conjecturer instantiate arbitrary game
scenarios — the TCG analogue of "write down a theorem statement."

---

## 0. The core idea (one paragraph)

SGS is **asymmetric self-play with a Guide**. One LLM plays three roles: a **Solver** (plays the game),
a **Conjecturer** (generates *simpler related game scenarios* that bridge the gap between what the Solver
can currently do and the hard target matchups it loses), and a frozen **Guide** (scores each generated
scenario by whether it is *on the path to* a target and *clean/non-degenerate*). The Solver is trained
with **REINFORCE_{1/2}** (only on scenarios it wins ≤ 50% of the time) so its entropy doesn't collapse and
keeps feeding the Conjecturer signal. The Conjecturer is trained on `R_synth = R_solve · R_guide`. The
Guide is what stops the Conjecturer from "hacking" the solve-rate reward by producing trivial auto-wins
(the TCG equivalent of the paper's disjunctive / over-long theorem collapse).

---

## 1. Concept mapping: Lean4 → Pokemon TCG

| SGS (paper, Lean4)                     | SGS-TCG (this repo)                                                                                   | Engine primitive |
|----------------------------------------|--------------------------------------------------------------------------------------------------------|------------------|
| Target problem `x` (hard theorem)      | A **hard scenario**: a matchup config + a decision-point state the Solver currently *loses*           | a losing decision point from `data/loser/lost_*.json` (has `search_begin_input` blob), or a self-play loss |
| Dataset `D` of targets                 | Curated set of hard scenarios across the 10 meta decks (`data/decks/deck_<slug>.json`)                | `data/decks/`, gauntlet results in `results/` |
| Synthetic problem `x̃` (a lemma)        | A **simpler related scenario**: an edit of the target that isolates a sub-skill                        | a `ScenarioSpec` (see §3) loaded via `search_begin` |
| Conjecturer `gϕ` writes `x̃`           | LLM emits a declarative **edit script** over the target state                                          | parsed → `search_begin` kwargs |
| Solver `πθ` writes a proof             | LLM **plays the game**: emits action index into `obs.select.option` each decision point              | `agent(obs) -> list[int]`, `cg/game.py:48` |
| Verifier (Lean compiler)               | The **cabt engine** returns `result ∈ {0,1,2}`                                                          | `state.result`, `runner.py:27` (`cg/api.py:376`) |
| Binary reward `v(y) ∈ {0,1}`           | `1` if Solver wins the game from the scenario, else `0`                                                | `cg/game.py:66` |
| `k` rollouts per problem               | `k` games per scenario (fix deck order + `manual_coin=True` for near-determinism, or sample for noise) | `search_begin(..., manual_coin=...)`, `cg/api.py:524` |
| Solve rate `s(x̃)`                     | Fraction of the `k` games the Solver wins from `x̃`                                                    | — |
| Guide `ρ` judges relevance + elegance  | LLM judges: is `x̃` on the path to mastering `x`, and is it a *clean, realistic* scenario?            | frozen LLM, SFT'd for format (§E.3 of paper) |
| REINFORCE_{1/2} Solver objective       | Train Solver only on scenarios with `s ≤ 0.5`                                                          | §G.2 of paper |
| Conjecturer collapse → disjunctive junk | Conjecturer collapse → **auto-win / degenerate states** (opp no basics, opp active at 1 HP, 10 energy t1) | the Guide downweights these |

---

## 2. The three roles, instantiated

All three are initialized from the **same base LLM** (as in the paper). In TCG the natural base is a
model with strong structured/JSON output; the existing heuristic agents (`agents/main.py`) are the
*behavioral priors* used to bootstrap `D` and to warm-start the Solver via SFT on their traces.

### 2.1 Solver `πθ` — the player
- **Input:** an observation rendered as a compact structured prompt: my hand, my active/bench (hp, energy,
  tools, statuses), opponent's visible board + counts, prize counts, turn, the *legal options* enumerated
  from `obs.select.option` (`cg/api.py:398-409`). Card identities resolved to names via `all_card_data()`
  (`cg/api.py:495`) so the LLM can reason about card text.
- **Output:** natural-language reasoning + a chosen option index. The action is the index into
  `select.option`; multi-select contexts (`minCount..maxCount`) return a list (the engine already accepts
  `list[int]`, `runner.py:74`).
- **Episode = one game.** Reward = terminal win (binary). Intermediate shaping optional but *keep the
  terminal signal dominant* (the paper uses pure binary verification).
- **Seat/global isolation:** the cabt agents store state in module globals (`plan`, `pre_turn`, see
  `runner.py:5-11`). The Solver MUST be seat-scoped — instantiate per seat, never rely on module globals,
  or the two sides corrupt each other.

### 2.2 Conjecturer `gϕ` — the scenario author
- **Input:** a target scenario `x` the Solver loses, rendered with a short "why this is hard" summary
  (e.g. "opponent has a Stage-2 Typhlosion with 3 {R} energy; you have no answer on board; you lose to
  its 160-dmg attack next turn").
- **Output:** a **simpler related scenario `x̃`** as an *edit script* — a declarative list of modifications
  to `x` that carve out a sub-skill, e.g.:
  - *reduce the threat:* "opponent's active is Quilava (Stage-1) instead of Typhlosion; 2 {R} energy"
  - *partial credit:* "start with one extra {R} attached to our active Crustle"
  - *isolate a tactic:* "pre-attach a Boss's Orders in hand; opponent active is gustable target"
  - *prune complexity:* "opponent bench reduced to 1 Pokémon"
- **Why edits, not from-scratch states:** generating a *legal* full TCG state from text is hard; editing a
  real target state keeps legality high and keeps `x̃` meaningfully "related to `x`" (the whole point of
  conditioning). This mirrors the paper conditioning the Conjecturer on the unsolved target.
- The edit script is translated (deterministic code) into `search_begin` kwargs and/or a `search_begin_input`
  blob, so the engine loads `x̃` and the Solver can play it.

### 2.3 Guide `ρ` — the frozen judge (the "self-guidance")
- **Frozen** (as in the paper). Optionally SFT'd for reliable output format (paper §4.1: 2048 GPT-4.1-mini
  examples raised well-formatted output from 54.7% → 99%).
- **Input:** `(target x, generated x̃)`.
- **Output:** a score `R_guide ∈ [0,1]` on (a) **relevance** — is `x̃` genuinely on the path to mastering
  `x`? and (b) **elegance/cleanness** — is it a realistic, non-degenerate, naturally-reachable state?
- The Guide is the mechanism that prevents the Conjecturer from collapsing to trivial auto-wins (see §6).

---

## 3. The scenario representation (the "theorem statement" analogue)

This is the central technical object. A `ScenarioSpec` is a declarative description of a loadable game
situation:

```
ScenarioSpec = {
  "matchup":   {"me": deck_slug, "opp": deck_slug, "opp_policy": "solver-checkpoint@v3"},
  "state":     {                     # a mid-game decision point
     "turn": int, "firstPlayer": int,
     "me":   {"hand":[cardIds], "active": PokemonSpec, "bench":[PokemonSpec], "prize":[...], "deck_order":[...]},
     "opp":  {"active": PokemonSpec|None, "bench":[...], "prize_count": int, "hand_count": int, "deck_order":[...]}
  },
  "edit_script": [ ... ],            # ONLY present for synthetic x̃; applied on top of a parent target
  "determinism": {"manual_coin": bool, "fixed_deck_order": bool}
}
```

- **Targets `x`** come from real games (so `state` is grounded and legal).
- **Synthetic `x̃`** carry an `edit_script` produced by the Conjecturer; the translator validates legality
  (basic Pokémon present, bench ≤ 5, energy counts sane, prize total = 6) before loading.
- **Loading into the engine** uses `search_begin` (`cg/api.py:517`): it takes the predicted
  `your_deck / your_prize / opponent_deck / opponent_prize / opponent_hand / opponent_active` plus the
  `search_begin_input` blob — exactly the controllable knobs of a scenario. The blobs already exist in
  `data/loser/lost_*.json`, giving a ready-made bootstrap `D`.

---

## 4. Algorithm (Algorithm 1 of the paper, adapted)

```
Require: target scenario set D, Solver πθ, Conjecturer gϕ, Guide ρ  (all init from same base LLM)
1:  Mark all x ∈ D unsolved
2:  for iteration t = 1, 2, ... do
3:      Sample batch B ⊆ D
4:      Split B into B_solved, B_unsolved                      # solved = Solver wins ≥ τ (e.g. 0.8) historically
5:      for each x ∈ B_unsolved do
6:          x̃ ← Conjecturer gϕ(· | x)                          # edit_script carving a sub-skill
7:          validate x̃ is legal; reject+resample if not
8:      Collect B_synth
9:      for each scenario z ∈ B ∪ B_synth do
10:         play k games: Solver πθ vs (opp policy), verify each with the engine → v ∈ {0,1}
11:         s(z) ← mean(v)
12:     for each x̃ ∈ B_synth do
13:         ind ← 1[ s(x̃) ≠ 0  AND  s(x̃) in bottom 70% of batch solve rates ]
14:         R_solve ← ind · (1 − s(x̃))
15:         R_guide ← ρ(x, x̃)
16:     Update Solver πθ with REINFORCE_{1/2} using v(y)        # §5
17:     Update Conjecturer gϕ with REINFORCE using R_synth = R_solve · R_guide   # batch-normalized to [0,1]
```

Notes vs. the paper:
- Line 10: "verify with the engine" replaces "verify with the Lean compiler." Verification is **stochastic**
  in TCG (shuffles/coins/flips), so we either fix them (`manual_coin=True`, fixed deck order) for a sharper
  signal or average over `k` rollouts (the paper uses `k=8` anyway).
- The opponent inside a scenario is a **checkpoint from the Solver league** — this is where classical
  symmetric self-play plugs in (the user's original framing). SGS itself is asymmetric (Conjecturer vs
  Solver); the in-game adversary is just part of the scenario spec.

---

## 5. Solver objective: REINFORCE_{1/2} (why this specifically)

Paper finding (§4.5): grouped objectives like CISPO suffer **entropy collapse** — the Solver becomes
near-deterministic, scenario solve rates concentrate at 0 and 1, and the Conjecturer is starved of reward
(0 and 1 solve rates both give `R_solve = 0`). This kills self-play.

`REINFORCE_{1/2}` trains **only on scenarios with solve rate ≤ 0.5** (paper §G.2). This (a) focuses
compute on still-hard scenarios and (b) keeps the Solver's policy entropy high, which keeps a healthy
spread of intermediate solve rates flowing to the Conjecturer. **Use this as the Solver objective**, with
an entropy bonus / KL-to-base regularizer if a grouped objective is ever desired.

Reward per rollout = binary win. Optionally apply the paper's length/Soft-Overlong penalty analogue — here,
a **step-count penalty** on rollouts that hit the engine step cap (`runner.py:40`, `max_steps`) to avoid
the stalling draws the existing heuristic agents already guard against (`main.py:188-189`).

---

## 6. Reward design & anti-collapse (the whole reason SGS works)

### 6.1 `R_solve` — difficulty signal (unchanged from paper)
```
ind    ← 1[ s(x̃) ≠ 0  ∧  s(x̃) in bottom 70% of batch solve rates ]
R_solve ← ind · (1 − s(x̃))
```
Favors scenarios of *intermediate difficulty*. Excludes unsolvable (`s=0`) and too-easy (top 30%).

### 6.2 `R_guide` — the self-guidance (rubric adapted to TCG)
The paper's rubric (§E.3): a 0–5 **relevance** score, a 0/1 **redundancy** flag, a 0–4 **conclusion
complexity** score; if complexity ≥ 3 the score is forced to 0, else
`R_guide = max(0, relevance + (2 − complexity) + (1 − redundancy))`, normalized to [0,1].

TCG adaptation:

- **Relevance (0–5):** is `x̃` on the critical path to mastering `x`?
  - 0: unrelated, OR identical to `x` (no simplification), OR an auto-win by rule.
  - 1: same deck type but the edit doesn't touch the actual difficulty of `x`.
  - 2: same archetype / similar board shape; tangentially useful.
  - 3: isolates a real sub-skill of `x` (e.g. the gust-and-KO line, the setup race).
  - 4: directly useful — mastering `x̃` clearly makes `x` easier.
  - 5: `x̃` is a faithful "lemma"; solving it dramatically reduces `x`'s difficulty.

- **Degeneracy (0–4) — the "complexity" analogue, inverted:** how *artificially stacked / trivially winnable*
  is `x̃`? (This is the exact TCG counterpart of the disjunctive-junk-theorem collapse the paper graphs in
  Fig. 2.)
  - 0: realistic, naturally reachable state.
  - 1: minor unrealistic conveniences.
  - 2: noticeably stacked (e.g. opponent at low HP with no retreat).
  - 3: heavily degenerate (opponent active at 1 HP; opponent prize count already 0 so you win by passing;
    10 energy on turn 1).
  - 4: an auto-win by construction (opponent has no Basic in active; impossible board).

- **Redundancy (0/1):** do the edits include modifications irrelevant to `x`'s difficulty (noise that
  doesn't relate to the target skill)?

- **Combine:** `R_guide = max(0, relevance + (2 − degeneracy) + (1 − redundancy))`, **forced to 0 if
  degeneracy ≥ 3** — exactly mirroring the paper's complexity≥3 → 0 rule. Normalize to [0,1].

This is the crux: **without the Guide**, `R_solve` alone rewards the Conjecturer for any winnable scenario,
and it will collapse to auto-wins that teach nothing (the paper's Fig. 6 "No Guide" ablation). The Guide is
what keeps synthetic scenarios grounded in the target.

### 6.3 `R_synth = R_solve · R_guide`, linearly normalized to [0,1] within each batch (paper §G.5).

---

## 7. Where `D` (the target set) comes from

1. **Loss replays:** `data/loser/lost_*.json` already contain losing decision points with
   `search_begin_input` blobs — instant grounded targets.
2. **Gauntlet losses:** `results/<slug>.txt` identify matchups the current best heuristic agent
   (`agents/main.py`) loses → those matchups' hard states are targets.
3. **Self-play losses:** as the Solver trains, games it loses against the checkpoint league are mined for
   new decision-point targets (this is the self-renewing curriculum — the TCG version of "epoching the
   target data 230×" in the paper).

Mark `x ∈ D` **solved** once the Solver's historical win rate on it ≥ τ (e.g. 0.8). Stop generating
synth for solved targets (paper line 4–5).

---

## 8. Engine-specific gotchas

- **Linux x86-64 only.** `libcg.so` runs under Docker emulation (README). The entire SGS loop
  (rollouts + verification) is engine-bound and slow → vectorize: many parallel containers, or precompute
  a large `(scenario, rollout, outcome)` replay buffer for offline Solver bootstrapping before closing the
  self-play loop.
- **Hidden information.** `search_begin` requires *guessed* `opponent_deck/hand/prize` (`cg/api.py:530-535`).
  The Solver should also emit a **belief** over these; the Conjecturer's edit scripts operate on the
  *believed* opponent state, and verification truth is the engine's actual resolution. This is the TCG
  equivalent of partial observability — handle it like the paper handles uncertainty: via averaged solve
  rates over `k` rollouts.
- **Shared module globals.** Heuristic agents keep `plan/pre_turn` at module scope (`runner.py:5-11`). The
  Solver MUST be seat-scoped (one policy instance per seat) — never reuse the module-global pattern.
- **Stalling/draws.** Existing agents cap actions per turn to avoid infinite-ability loops (`main.py:188`).
  Carry this into the Solver's env wrapper + use the step-count penalty (§5) so draws don't pollute the
  binary reward.

---

## 9. Suggested repo layout

```
rl/
├── scenario.py        # ScenarioSpec + edit_script → search_begin kwargs (legality-checked)
├── env.py             # Gym-style wrapper: battle_start/select, pointer action over select.option,
│                      #   seat-scoped Solver, binary win reward, step-cap penalty
├── targets.py         # build D from data/loser/*.json + gauntlet + self-play losses; track solved set
├── roles/
│   ├── solver.py      # πθ: obs(prompt) → option index (+NL reasoning). REINFORCE_{1/2} trainer.
│   ├── conjecturer.py # gϕ: target(prompt) → edit_script. REINFORCE on R_synth trainer.
│   └── guide.py       # ρ: frozen judge; SFT'd for format; R_guide rubric (§6.2)
├── rewards.py         # R_solve, R_guide combine, batch normalization
├── verify.py          # run k games via search_begin/search_step; return solve rate
├── league.py          # checkpoint pool of past Solvers (in-game adversaries + eval)
├── sgs_loop.py        # Algorithm 1 driver (§4)
└── prompts/           # conjecturer.txt, guide.txt, solver.txt (mirrors of paper §E)
```

## 10. Bootstrap order (smallest useful first step)

1. `rl/scenario.py` + `rl/verify.py` — turn `data/loser/lost_*.json` into loadable scenarios and confirm
   you can replay/verify a win from an edited state via `search_begin`. **This de-risks everything.**
2. `rl/roles/guide.py` prompt + rubric (§6.2) — get the frozen judge producing parseable `R_guide`.
3. Warm-start the Solver via SFT on traces from `agents/main.py`, then turn on REINFORCE_{1/2} on `D` only
   (this is the "RL baseline" curve in the paper's Fig. 4).
4. Add the Conjecturer + Guide to close the SGS loop (§4).

---

## Reference: paper ↔ this doc

- Algorithm 1 → §4
- Solver REINFORCE_{1/2} (§G.2) → §5
- Conjecturer reward & normalization (§3.2, §G.5) → §6
- Guide rubric (§E.3) → §6.2
- Ablations: No-Guide / Frozen-Conjecturer / No-Conditioning (§4.4), entropy/REINFORCE_{1/2} (§4.5) → §5, §6
- Extension to non-verifiable / embodied domains (§5 Limitations) → this is that extension.
