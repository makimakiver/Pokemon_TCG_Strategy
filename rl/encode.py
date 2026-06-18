"""Observation + per-option featurizer for the Solver pointer net.

Deck-agnostic: every feature is read from the runtime card/attack tables, so the
same encoder works for any deck (mirrors ``agents/bare_agent.py``). Two outputs:

* ``featurize(obs)`` -> (global_feats[G], option_feats[N, F], legal_mask[N])
* ``option_prior_scores(obs)`` -> heuristic logits[N], the scripted prior the
  policy adds as an annealed residual and that ``sft.py`` clones.

Imports ``cg`` (Linux-only native lib) -> runs inside the Docker engine image.
"""
from __future__ import annotations

import numpy as np

from cg.api import (
    AreaType, CardType, EnergyType, OptionType, Pokemon, SelectContext,
    all_attack, all_card_data,
)
from .config import CONFIG

# --- Runtime card / attack databases ---------------------------------------
_all_card = all_card_data()
CARD_TABLE = {c.cardId: c for c in _all_card}
try:
    ATTACK_TABLE = {a.attackId: a for a in all_attack()}
except Exception:  # pragma: no cover - engine may not expose attacks
    ATTACK_TABLE = {}

# Feature widths must match config.option_feat_dim / global_feat_dim.
OPTION_FEATURE_NAMES = [
    # one-hot-ish option type bucket (8)
    "is_play", "is_attach", "is_evolve", "is_ability", "is_attack",
    "is_retreat", "is_end", "is_card_select",
    # target card stats (when the option references a card) (12)
    "card_is_pokemon", "card_is_basic", "card_is_stage1", "card_is_stage2",
    "card_is_energy", "card_is_item", "card_is_supporter", "card_is_tool",
    "card_hp", "card_retreat", "card_is_ex", "card_prize_value",
    # attack stats (4)
    "attack_damage", "attack_lethal", "attack_weakness_hit", "attack_cost",
    # board / hand context for the option (8)
    "opt_player_is_me", "opt_area_active", "opt_area_bench", "opt_area_hand",
    "energy_on_target", "tools_on_target", "context_id", "prior_score",
]
assert len(OPTION_FEATURE_NAMES) == CONFIG.option_feat_dim, (
    len(OPTION_FEATURE_NAMES), CONFIG.option_feat_dim)


def _stage(cid: int) -> int:
    c = CARD_TABLE.get(cid)
    if c is None:
        return 0
    return 2 if getattr(c, "stage2", False) else 1 if getattr(c, "stage1", False) else 0


def _best_attack(cid: int):
    c = CARD_TABLE.get(cid)
    if c is None:
        return None
    best, best_dmg = None, -1
    for a in (c.attacks or []):
        atk = ATTACK_TABLE.get(a)
        d = getattr(atk, "damage", 0) or 0
        if d > best_dmg:
            best, best_dmg = atk, d
    return best


def _prize_value(cid: int) -> int:
    c = CARD_TABLE.get(cid)
    if c is None:
        return 1
    return 3 if getattr(c, "megaEx", False) else 2 if getattr(c, "ex", False) else 1


def get_card(obs, area, index, player_index):
    """Resolve the Card/Pokemon an option points at. Mirrors bare_agent.get_card."""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except (IndexError, TypeError, AttributeError):
        return None
    return None


def _card_of_option(obs, o, my_index):
    """Best-effort resolve the card an option references (hand for PLAY/ATTACH)."""
    if o.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, o.index, my_index)
    if o.type in (OptionType.ATTACH, OptionType.EVOLVE):
        return get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
    if o.area is not None and o.index is not None:
        return get_card(obs, o.area, o.index,
                        o.playerIndex if o.playerIndex is not None else my_index)
    return None


def featurize_global(obs) -> np.ndarray:
    """Per-state features shared by every option (width == config.global_feat_dim)."""
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    g = np.zeros(CONFIG.global_feat_dim, dtype=np.float32)
    g[0] = st.turn / 50.0
    g[1] = st.turnActionCount / 50.0
    g[2] = float(st.supporterPlayed)
    g[3] = float(st.energyAttached)
    g[4] = float(st.retreated)
    g[5] = len(me.prize) / 6.0
    g[6] = len(op.prize) / 6.0
    g[7] = me.handCount / 20.0
    g[8] = me.deckCount / 60.0
    g[9] = op.deckCount / 60.0
    g[10] = len(me.bench) / 5.0
    g[11] = len(op.bench) / 5.0
    my_active = me.active[0] if me.active else None
    op_active = op.active[0] if op.active else None
    if my_active:
        g[12] = my_active.hp / 340.0
        g[13] = len(my_active.energies) / 6.0
    if op_active:
        g[14] = op_active.hp / 340.0
        g[15] = len(op_active.energies) / 6.0
    g[16] = float(me.poisoned)
    g[17] = float(me.burned)
    g[18] = float(me.asleep or me.paralyzed or me.confused)
    g[19] = float(obs.select.context) / 48.0 if obs.select else 0.0
    g[20] = obs.select.minCount / 6.0 if obs.select else 0.0
    g[21] = obs.select.maxCount / 6.0 if obs.select else 0.0
    g[22] = len(obs.select.option) / 30.0 if obs.select else 0.0
    g[23] = 1.0
    return g


def featurize_option(obs, o, my_index, prior_score: float) -> np.ndarray:
    f = np.zeros(CONFIG.option_feat_dim, dtype=np.float32)
    t = o.type
    f[0] = float(t == OptionType.PLAY)
    f[1] = float(t == OptionType.ATTACH)
    f[2] = float(t == OptionType.EVOLVE)
    f[3] = float(t == OptionType.ABILITY)
    f[4] = float(t == OptionType.ATTACK)
    f[5] = float(t == OptionType.RETREAT)
    f[6] = float(t == OptionType.END)
    f[7] = float(t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD,
                       OptionType.ENERGY))

    card = _card_of_option(obs, o, my_index)
    cid = getattr(card, "id", None)
    cd = CARD_TABLE.get(cid) if cid is not None else None
    if cd is not None:
        f[8] = float(cd.cardType == CardType.POKEMON)
        f[9] = float(_stage(cid) == 0 and cd.cardType == CardType.POKEMON)
        f[10] = float(getattr(cd, "stage1", False))
        f[11] = float(getattr(cd, "stage2", False))
        f[12] = float(cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY))
        f[13] = float(cd.cardType == CardType.ITEM)
        f[14] = float(cd.cardType == CardType.SUPPORTER)
        f[15] = float(cd.cardType == CardType.TOOL)
        f[16] = (cd.hp or 0) / 340.0
        f[17] = (cd.retreatCost or 0) / 5.0
        f[18] = float(getattr(cd, "ex", False) or getattr(cd, "megaEx", False))
        f[19] = _prize_value(cid) / 3.0

    # Attack stats for ATTACK options.
    if t == OptionType.ATTACK and o.attackId in ATTACK_TABLE:
        atk = ATTACK_TABLE[o.attackId]
        dmg = atk.damage or 0
        f[20] = dmg / 340.0
        op = obs.current.players[1 - my_index]
        op_active = op.active[0] if op.active else None
        if op_active is not None:
            f[21] = float(dmg >= op_active.hp)
            mine = obs.current.players[my_index].active
            mtype = CARD_TABLE[mine[0].id].energyType if mine and mine[0] and mine[0].id in CARD_TABLE else None
            tdata = CARD_TABLE.get(op_active.id)
            f[22] = float(tdata is not None and mtype is not None and tdata.weakness == mtype)
        f[23] = len(atk.energies or []) / 5.0

    # Board context.
    f[24] = float(o.playerIndex == my_index) if o.playerIndex is not None else 1.0
    f[25] = float(o.inPlayArea == AreaType.ACTIVE or o.area == AreaType.ACTIVE)
    f[26] = float(o.inPlayArea == AreaType.BENCH or o.area == AreaType.BENCH)
    f[27] = float(o.area == AreaType.HAND)
    if isinstance(card, Pokemon):
        f[28] = len(card.energies) / 6.0
        f[29] = len(card.tools) / 3.0
    f[30] = float(obs.select.context) / 48.0 if obs.select else 0.0
    f[31] = prior_score
    return f


def featurize(obs):
    """Return (global[G], options[N,F], mask[N]) for the current select.

    ``mask`` is all-ones here: the engine only ever offers legal options, and the
    micro-step env handles the variable-length subset selection itself.
    """
    if obs.select is None:
        return (np.zeros(CONFIG.global_feat_dim, np.float32),
                np.zeros((0, CONFIG.option_feat_dim), np.float32),
                np.zeros(0, np.float32))
    my_index = obs.current.yourIndex
    priors = option_prior_scores(obs)
    pr = _softmax(priors)
    g = featurize_global(obs)
    opts = np.stack([featurize_option(obs, o, my_index, float(pr[i]))
                     for i, o in enumerate(obs.select.option)], axis=0)
    mask = np.ones(len(obs.select.option), dtype=np.float32)
    return g, opts, mask


def _softmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


# --- Scripted option prior (distilled from bare_agent / main_v1) ------------
# Coarse priorities matching bare_agent's score buckets. Used both as the
# annealed policy residual and as the SFT target. Kept intentionally simple:
# the net is meant to *surpass* it, not match it exactly.
def option_prior_scores(obs) -> np.ndarray:
    sel = obs.select
    n = len(sel.option)
    scores = np.zeros(n, dtype=np.float32)
    if n == 0:
        return scores
    st = obs.current
    my_index = st.yourIndex
    op = st.players[1 - my_index]
    op_prize_left = len(op.prize)
    op_active = op.active[0] if op.active else None

    for i, o in enumerate(sel.option):
        t = o.type
        s = 0.0
        if t == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            cd = CARD_TABLE.get(getattr(card, "id", None))
            if cd is not None:
                s = {CardType.POKEMON: 6.0, CardType.ITEM: 4.5, CardType.SUPPORTER: 4.0,
                     CardType.STADIUM: 3.0}.get(cd.cardType, 2.0)
        elif t == OptionType.ATTACK:
            atk = ATTACK_TABLE.get(o.attackId)
            dmg = getattr(atk, "damage", 0) or 0
            s = 3.0 + dmg / 100.0
            if op_active is not None and dmg >= op_active.hp:
                s += 3.0
                if op_prize_left <= _prize_value(op_active.id):
                    s += 10.0
        elif t == OptionType.EVOLVE:
            s = 5.0
        elif t == OptionType.ATTACH:
            s = 4.0
        elif t == OptionType.ABILITY:
            s = 3.5
        elif t == OptionType.RETREAT:
            s = 1.0
        elif t == OptionType.YES:
            s = 1.0
        elif t == OptionType.NO:
            s = 0.5
        elif t == OptionType.NUMBER:
            s = (o.number or 0) / 10.0
        elif t == OptionType.END:
            s = 0.2
        else:  # CARD / ENERGY selections: mild preference, handled by net
            s = 1.5
        scores[i] = s
    return scores
