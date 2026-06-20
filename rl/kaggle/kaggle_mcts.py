"""Kaggle MCTS self-play trainer, ported to run locally against the repo's cg engine.

This is the Kaggle notebook script verbatim EXCEPT for two host-portability changes:

  1. Import bootstrap. Kaggle locates the engine with
         sys.path.append(glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0])
     Locally `cg` is a package at the repo root, so we just make sure the repo
     root is importable (it already is when run as `python -m rl.kaggle.kaggle_mcts`
     from /app; the fallback below also covers `python rl/kaggle_mcts.py`).

  2. Loop sizes are env-overridable so you can smoke-test cheaply. The defaults
     match the Kaggle script exactly:
         RL_OUTER_ITERS    (default 5)    outer save/eval/selfplay/train rounds
         RL_EVAL_GAMES     (default 50)   eval games per round (MCTS vs random)
         RL_SELFPLAY_GAMES (default 100)  self-play games per round
         RL_SEARCH_COUNT   (default 10)   MCTS simulations per decision

`cg/libcg.so` is Linux x86-64 only, so this must run inside the cabt-rl image:

    docker build --platform=linux/amd64 -f rl/Dockerfile -t cabt-rl .

    # full Kaggle-equivalent run
    docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
      --entrypoint python cabt-rl -m rl.kaggle_mcts

    # cheap smoke test (1 round, 2 eval + 2 self-play games, 4 sims)
    docker run --rm --platform=linux/amd64 -v "$PWD":/app -w /app \
      -e RL_OUTER_ITERS=1 -e RL_EVAL_GAMES=2 -e RL_SELFPLAY_GAMES=2 \
      -e RL_SEARCH_COUNT=4 --entrypoint python cabt-rl -m rl.kaggle_mcts

Model checkpoints are written to ./out/model{0..N}.pth (mounted back to the host).
"""
import glob
import json
import math
import os
import random
import sys

import torch
import torch.nn
import torch.nn.functional
import torch.optim

# --- Local engine bootstrap (replaces the Kaggle /kaggle/input glob) ---------
# Prefer an already-importable `cg` (the repo root, e.g. /app in the cabt-rl
# image). Fall back to a glob for a vendored cg-lib if someone drops one in.
try:
    import cg.api  # noqa: F401  (probe: is the engine already on the path?)
except ModuleNotFoundError:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    try:
        import cg.api  # noqa: F401
    except ModuleNotFoundError:
        hits = glob.glob("/kaggle/input/**/cg-lib", recursive=True)
        if hits:
            sys.path.append(hits[0])
        else:
            raise

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import battle_start, battle_finish, battle_select

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1 # Max Card ID + 1

attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1 # Max Attack ID + 1

num_words_encoder = 24
encoder_size = 22000 # Encoder input size exceeding the vocabulary size

decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count # Decoder input vocabulary size

SEARCH_COUNT = int(os.environ.get("RL_SEARCH_COUNT", "10")) # MCTS Search count

# Decoder Layer of MyModel
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()

        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)

# My Transformer Model
class MyModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int
    ):
        super(MyModel, self).__init__()

        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)

# torch.nn.EmbeddingBag input
class SparseVector:
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))

# Add encoder card feature
def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card != None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)

# Add encoder cards feature
def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards != None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)

# Add encoder Pokémon feature
def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke == None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)

# Add encoder player feature
def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)

def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8): # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

# Add decoder Main Select feature
def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card != None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)

# Add decoder Card ID feature
def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)

# Add decoder Card feature
def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card != None:
        decoder_card_id(sv, context, card.id)

def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv

# Evaluate with My Model
def eval_nn(sv_enc: SparseVector, sv_dec:SparseVector, model: MyModel) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device))

    return (value.tolist()[0][0], policy.tolist()[0])

# Single Training Sample
class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec:SparseVector):
        self.value = value # Encoder output
        self.policy = policy # Decoder output
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec

# MCTS Node Child
class Child:
    node: 'Node | None'
    select: list[int] # Selected option indices
    prob: float # Probability

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob

# MCTS Node
class Node:
    value: float # Self value
    total: float # Total value
    visit: int # Visit count
    parent: 'Node | None' # Parent node
    children: list[Child]
    state: SearchState # Search State of this node

    def __init__(self, parent: 'Node | None', state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    # Backpropagation value
    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent != None:
            self.parent.backprop(value)

def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel
    ) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        sample = None
    else:
        actions = []
        indices = list(range(obs.select.maxCount))
        for _ in range(64):
            actions.append(indices.copy())
            for i in range(len(indices)):
                index = len(indices) - i - 1
                if indices[index] < len(obs.select.option) - i - 1:
                    indices[index] += 1
                    for j in range(index+1, len(indices)):
                        indices[j] = indices[j - 1] + 1
                    break
            else:
                break

        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            sum += p
        for c in node.children:
            c.prob /= sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)

# We will perform exploration using MCTS and select actions. At the same time, we will also generate training data.
def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, state.players[your_index].deckCount), # Randomly select from deck.
        your_prize=random.sample(your_deck, len(state.players[your_index].prize)), # Randomly select from deck.
        opponent_deck=[1072] * state.players[1 - your_index].deckCount, # Fill with Snorlax (There is no deep meaning).
        opponent_prize=[1] * len(state.players[1 - your_index].prize), # Fill with Basic Energy (There is no deep meaning)
        opponent_hand=[1] * state.players[1 - your_index].handCount, # Fill with Basic Energy.
        opponent_active=[1072] if len(active) > 0 and active[0] == None else []) # Fill with Snorlax.
    root, sample = create_node(None, search_state, your_index, your_deck, model) # Create root node.

    # Search
    for _ in range(SEARCH_COUNT):
        current = root
        while True:
            value = -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node == None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next = child

            if next.node == None:
                search_state = search_step(current.state.searchId, next.select)
                next.node, _ = create_node(current, search_state, your_index, your_deck, model)
                break
            else:
                current = next.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    # Select the most visited node.
    max_child = None
    max_visit = -1
    min_value = 10
    for child in root.children:
        if child.node != None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    # Generate training data
    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node == None:
            v = min_value - v - 0.03
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return (max_child.select, sample)


# Helper class to construct batch inputs for the neural network.
class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)

# Opponent for evaluation.
def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount) # Select at random.

# For displaying progress.
def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if(current >= count):
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1

# A sample deck for training.
sample_deck = [721,721,722,722,722,722,723,723,723,723,1092,1121,1121,1145,1145,1163,1163,1219,1219,1219,1219,1227,1227,1227,1227,1262,1262,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

# Optional: train the net on a DIFFERENT deck (RL_DECK = path to a 60-card JSON
# list), e.g. the Honchkrow 26267 list. The net is deck-conditioned, so it must be
# trained on whatever deck it will ship with.
_deck_override = os.environ.get("RL_DECK", "")
if _deck_override and os.path.exists(_deck_override):
    sample_deck = json.load(open(_deck_override))
    assert len(sample_deck) == 60, f"RL_DECK has {len(sample_deck)} cards"
    print(f"[deck] training on RL_DECK={_deck_override} (first ids {sample_deck[:6]})", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(128, 2, 256, 1, 1)
model = model.to(device)
# Optional warm start: initialize from a saved checkpoint instead of random
# weights (e.g. RL_INIT_CKPT=out/random_run/model2.pth to continue the round-2 net).
_init_ckpt = os.environ.get("RL_INIT_CKPT", "")
if _init_ckpt:
    model.load_state_dict(torch.load(_init_ckpt, map_location=device))
    print(f"[init] warm-started from {_init_ckpt}", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn_enc = torch.nn.HuberLoss(delta=0.2)  # Encoder loss function
loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)  # Decoder loss function
os.makedirs("out", exist_ok=True)

# Outer/eval/self-play loop sizes (env-overridable; defaults match Kaggle).
OUTER_ITERS = int(os.environ.get("RL_OUTER_ITERS", "5"))
EVAL_GAMES = int(os.environ.get("RL_EVAL_GAMES", "50"))
SELFPLAY_GAMES = int(os.environ.get("RL_SELFPLAY_GAMES", "100"))

# Per-round eval win rates, dumped to out/metrics.json for rl/performance.html.
metrics = []

# --- Opponents: decoupled eval metric vs self-play data generation -----------
# RL_EVAL_OPP / RL_SELFPLAY_OPP accept:
#   "self"    -> MCTS vs MCTS self-play (for eval, scored against random_agent).
#   "random"  -> random_agent.
#   "bare"    -> agents.bare_agent forced onto a MIRROR sample_deck.
#   "<name>"  -> any agents.<name> module (e.g. "main_v5") playing its OWN my_deck.
#   "mix"     -> (RL_SELFPLAY_OPP only) mostly self-play + RL_MIX_FRAC of games vs
#                the EVAL opponent — keeps the target opponent IN the data without
#                the all-loss value-collapse a pure-opponent diet causes.
# RL_MIX_FRAC: fraction of self-play games played vs the eval opponent when "mix".
# (Back-compat: RL_OPPONENT seeds both if the new vars aren't set.)
import importlib

_legacy = os.environ.get("RL_OPPONENT")
RL_EVAL_OPP = os.environ.get("RL_EVAL_OPP", _legacy or "self")
RL_SELFPLAY_OPP = os.environ.get("RL_SELFPLAY_OPP", _legacy or "self")
RL_MIX_FRAC = float(os.environ.get("RL_MIX_FRAC", "0.25"))
_bare_every = max(1, round(1.0 / RL_MIX_FRAC)) if RL_MIX_FRAC > 0 else 0


def _resolve_opponent(spec):
    """spec -> (move_fn(obs_dict)->list[int], opp_deck). opp_deck == sample_deck is
    a mirror; a different list means the opponent pilots its own deck."""
    if spec in ("self", "random"):
        return (random_agent, list(sample_deck))
    if spec == "bare":
        # generic pilot forced onto a MIRROR sample_deck (BARE_DECK set pre-import).
        with open("out/sample_deck.json", "w") as f:
            json.dump(sample_deck, f)
        os.environ["BARE_DECK"] = "out/sample_deck.json"
        mod = importlib.import_module("agents.bare_agent")
        return (mod.agent, list(sample_deck))
    # any other agent module, playing its OWN deck (e.g. main_v5 -> Honchkrow 26267).
    mod = importlib.import_module(spec if "." in spec else f"agents.{spec}")
    return (mod.agent, list(mod.my_deck))


def start_game(net_seat, opp_deck):
    """Start a battle: net's sample_deck at net_seat, opp_deck on the other seat."""
    return battle_start(sample_deck, opp_deck) if net_seat == 0 else battle_start(opp_deck, sample_deck)


# Eval opponent (the win rate is measured against this).
opponent_move, EVAL_OPP_DECK = _resolve_opponent(RL_EVAL_OPP)

# Self-play opponent for the non-self portion of data.
if RL_SELFPLAY_OPP == "self":
    SP_OPP_MOVE, SP_OPP_DECK = None, None
elif RL_SELFPLAY_OPP == "mix":
    SP_OPP_MOVE, SP_OPP_DECK = opponent_move, EVAL_OPP_DECK  # mix in the eval opponent
else:
    SP_OPP_MOVE, SP_OPP_DECK = _resolve_opponent(RL_SELFPLAY_OPP)

print(f"[opponent] eval vs {RL_EVAL_OPP}; self-play data = {RL_SELFPLAY_OPP}"
      + (f" (1 in {_bare_every} games vs {RL_EVAL_OPP})" if RL_SELFPLAY_OPP == "mix" else ""),
      flush=True)

# The main training loop.
for counter in range(OUTER_ITERS):
    torch.save(model.state_dict(), "out/model" + str(counter) + ".pth")  # Save the current model.
    sample_list:list[LearnSample] = []  # List of training data samples.

    model.eval()
    with torch.inference_mode():
        # Evaluation
        results = [0, 0, 0]

        for i in progress(EVAL_GAMES, "Evaluating... "):
            your_index = i % 2
            obs, start_data = start_game(your_index, EVAL_OPP_DECK)
            if start_data.errorPlayer >= 0:
                error = "Deck error."
                if start_data.errorType == 1:
                    error = "The deck contains invalid card ID."
                elif start_data.errorType == 2:
                    error = "You can include up to four cards with the same name in the deck, excluding basic Energy cards."
                elif start_data.errorType == 3:
                    error = "There are no Basic Pokémon in the deck."
                elif start_data.errorType == 4:
                    error = "You can include only one Ace Spec card in the deck."
                raise ValueError(error)
            while True:
                # Break the loop if the game has ended.
                if obs["current"]["result"] >= 0:
                    break

                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, sample_deck, model)
                else:
                    selected = opponent_move(obs)
                obs = battle_select(selected)

            battle_finish()  # Finalize the game.

            if obs["current"]["result"] == 2:  # Draw
                results[2] += 1
            elif obs["current"]["result"] == your_index:  # Win
                results[0] += 1
            else: # Lose
                results[1] += 1
        win_rate = 100 * results[0] // (results[0] + results[1])
        print("Evaluation win rate " + str(win_rate) + "%", flush=True)
        metrics.append({
            "round": counter,
            "win_rate": win_rate,
            "wins": results[0],
            "losses": results[1],
            "draws": results[2],
            "eval_games": EVAL_GAMES,
            "opponent": RL_EVAL_OPP,
            "selfplay_opp": RL_SELFPLAY_OPP,
        })
        with open("out/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Self Play (data collection). LAMBDA blends the final win/loss outcome
        # with each step's search value for the regression target (TD(lambda)-ish).
        LAMBDA = 0.9

        def label_trajectory(traj, won):
            # Walk a seat's samples backwards, bootstrapping value labels.
            value = 1.0 if won else -1.0
            for sample in reversed(traj):
                sample.value = (value + sample.value) * 0.5
                value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                sample_list.append(sample)

        for g in progress(SELFPLAY_GAMES, "Training Data Collecting... "):
            # Decide this game's data source: self-play (MCTS vs MCTS) or vs-opponent.
            if SP_OPP_MOVE is None:               # pure self-play
                opp_game = False
            elif RL_SELFPLAY_OPP == "mix":        # mostly self-play, some vs opponent
                opp_game = (_bare_every > 0 and g % _bare_every == 0)
            else:                                  # pure vs opponent
                opp_game = True

            if not opp_game:
                # MCTS vs MCTS — collect a trajectory for BOTH seats (balanced).
                obs, _ = battle_start(sample_deck, sample_deck)
                samples:list[list[LearnSample]] = [[], []]
                while obs["current"]["result"] < 0:
                    selected, sample = mcts_agent(obs, sample_deck, model)
                    samples[obs["current"]["yourIndex"]].append(sample)
                    obs = battle_select(selected)
                battle_finish()
                result = obs["current"]["result"]
                for i in range(2):
                    label_trajectory(samples[i], won=(i == result))
            else:
                # MCTS net vs the scripted opponent (its own deck) — net seat only.
                net_seat = g % 2  # alternate seats so the net learns both sides
                obs, _ = start_game(net_seat, SP_OPP_DECK)
                net_samples = []
                while obs["current"]["result"] < 0:
                    if obs["current"]["yourIndex"] == net_seat:
                        selected, sample = mcts_agent(obs, sample_deck, model)
                        net_samples.append(sample)
                    else:
                        selected = SP_OPP_MOVE(obs)
                    obs = battle_select(selected)
                battle_finish()
                label_trajectory(net_samples, won=(net_seat == obs["current"]["result"]))

    # Train on the training data collected through self-play.
    print("Training Start.")
    model.train()
    random.shuffle(sample_list)
    BATCH_SIZE = 128
    batch_count = len(sample_list) // BATCH_SIZE
    for i in range(batch_count):
        # Prepare a batch of data.
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask = []
        label_enc = []
        label_dec = []
        start = BATCH_SIZE * i
        for j in range(start, start + BATCH_SIZE):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(64 - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        # Convert data to PyTorch tensors.
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
        mask_tensor = mask_tensor.view(BATCH_SIZE, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device)
        label_tensor_enc = label_tensor_enc.view(BATCH_SIZE, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device)
        label_tensor_dec = label_tensor_dec.view(BATCH_SIZE, -1)

        optimizer.zero_grad()

        # Get model predictions for the batch.
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device))

        # Calculate loss.
        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = loss_dec * mask_tensor
        loss_dec = loss_dec.sum() / float(BATCH_SIZE)
        loss = loss_enc + loss_dec

        # Backpropagate the loss and update model parameters.
        loss.backward()
        optimizer.step()
    print("Training Finish.")
