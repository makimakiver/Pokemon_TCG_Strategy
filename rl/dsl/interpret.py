# rl/dsl/interpret.py
"""Compile a rule-set into a sim agent. Engine side (imports cg via rl.encode)."""
from __future__ import annotations

import json

from cg.api import (
    AreaType, CardType, OptionType, Pokemon, SelectContext, to_observation_class,
)
from rl.engine import encode
from rl.dsl.grammar import validate
from rl.dsl.predicates import (
    CardInfo, AttackInfo, OptInfo, RuleContext, score_options, pick_from_scores,
)

_CTX_MAP = {
    int(SelectContext.MAIN): "main",
    int(SelectContext.SETUP_ACTIVE_POKEMON): "setup",
    int(SelectContext.SETUP_BENCH_POKEMON): "setup",
    int(SelectContext.TO_HAND): "to_hand",
    int(SelectContext.ATTACH_FROM): "attach_from",
    int(SelectContext.DISCARD): "discard",
    int(SelectContext.SWITCH): "switch",
    int(SelectContext.DRAW_COUNT): "choose_count",
    int(SelectContext.DAMAGE_COUNTER_COUNT): "choose_count",
}

_KIND = {
    int(OptionType.ATTACK): "attack", int(OptionType.EVOLVE): "evolve",
    int(OptionType.ABILITY): "ability", int(OptionType.RETREAT): "retreat",
    int(OptionType.END): "end", int(OptionType.YES): "yes", int(OptionType.NO): "no",
    int(OptionType.NUMBER): "number",
    int(OptionType.CARD): "card_select", int(OptionType.TOOL_CARD): "card_select",
    int(OptionType.ENERGY_CARD): "card_select", int(OptionType.ENERGY): "card_select",
}


def _card_info(cid, pokemon) -> CardInfo:
    cd = encode.CARD_TABLE.get(cid)
    if cd is None:
        return CardInfo()
    stage = encode._stage(cid)
    is_poke = cd.cardType == CardType.POKEMON
    return CardInfo(
        is_pokemon=is_poke,
        is_basic=is_poke and stage == 0,
        is_stage1=bool(getattr(cd, "stage1", False)),
        is_stage2=bool(getattr(cd, "stage2", False)),
        is_ex=bool(getattr(cd, "ex", False) or getattr(cd, "megaEx", False)),
        is_energy=cd.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY),
        is_item=cd.cardType == CardType.ITEM,
        is_supporter=cd.cardType == CardType.SUPPORTER,
        is_tool=cd.cardType == CardType.TOOL,
        is_attacker=encode._best_attack(cid) is not None and (
            stage >= 1 or (getattr(encode._best_attack(cid), "damage", 0) or 0) >= 90),
        hp=cd.hp or 0,
        energy_count=len(pokemon.energies) if isinstance(pokemon, Pokemon) else 0,
    )


def _play_kind(cid) -> str:
    cd = encode.CARD_TABLE.get(cid)
    if cd is None:
        return "other"
    return {CardType.POKEMON: "play_pokemon", CardType.ITEM: "play_item",
            CardType.SUPPORTER: "play_supporter", CardType.STADIUM: "play_stadium"}.get(
                cd.cardType, "other")


def build_context(obs) -> RuleContext:
    st = obs.current
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]
    op_active = op.active[0] if op.active else None
    hand = me.hand or []
    has_energy = any(encode.CARD_TABLE.get(c.id) and encode.CARD_TABLE[c.id].cardType
                     in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY) for c in hand)
    bench_attacker = any(encode._best_attack(p.id) for p in me.bench)
    return RuleContext(
        my_prizes=len(me.prize), opp_prizes=len(op.prize),
        opp_active_hp=op_active.hp if op_active else 999,
        energy_unused=not st.energyAttached, supporter_unused=not st.supporterPlayed,
        bench_count=len(me.bench), bench_has_attacker=bench_attacker,
        hand_has_energy=has_energy, my_active_at_risk=False,
        context=_CTX_MAP.get(int(obs.select.context), "other"),
    )


def build_optinfo(obs, o) -> OptInfo:
    my_index = obs.current.yourIndex
    t = int(o.type)
    kind = _KIND.get(t, "other")
    if t == int(OptionType.PLAY):
        card = encode.get_card(obs, AreaType.HAND, o.index, my_index)
        kind = _play_kind(card.id) if card is not None else "other"
    elif t == int(OptionType.ATTACH):
        # For ATTACH, o.area/o.index identify the card BEING attached (energy/tool,
        # normally in HAND); o.inPlayArea/o.inPlayIndex identify the target Pokemon
        # (resolved below via _card_of_option). Kind is decided by the attached card.
        attach_area = o.area if o.area is not None else AreaType.HAND
        card = encode.get_card(obs, attach_area, o.index, my_index)
        cd = encode.CARD_TABLE.get(getattr(card, "id", None))
        kind = "attach_tool" if cd and cd.cardType == CardType.TOOL else "attach_energy"

    card = encode._card_of_option(obs, o, my_index)
    target = None
    is_mine = True
    if card is not None:
        target = _card_info(card.id, card)
        if o.playerIndex is not None:
            is_mine = (o.playerIndex == my_index)

    attack = None
    if t == int(OptionType.ATTACK) and o.attackId in encode.ATTACK_TABLE:
        atk = encode.ATTACK_TABLE[o.attackId]
        op = obs.current.players[1 - my_index]
        op_active = op.active[0] if op.active else None
        dmg = atk.damage or 0
        mine = obs.current.players[my_index].active
        mtype = (encode.CARD_TABLE[mine[0].id].energyType
                 if mine and mine[0] and mine[0].id in encode.CARD_TABLE else None)
        tdata = encode.CARD_TABLE.get(op_active.id) if op_active else None
        attack = AttackInfo(
            damage=dmg,
            lethal=bool(op_active and dmg >= op_active.hp),
            hits_weakness=bool(tdata and mtype is not None and tdata.weakness == mtype),
            affordable=True,
        )
    return OptInfo(kind=kind, is_mine=is_mine, target=target, attack=attack)


def compile(ruleset: dict):
    ok, errors = validate(ruleset)
    if not ok:
        raise ValueError("invalid ruleset: " + "; ".join(errors))

    def agent(obs_dict, _deck=None):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _deck if _deck is not None else []
        sel = obs.select
        ctx = build_context(obs)
        opts = [build_optinfo(obs, o) for o in sel.option]
        scores = score_options(ruleset, ctx, opts)
        return pick_from_scores(scores, sel.minCount, sel.maxCount)

    return agent


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
