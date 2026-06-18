"""Analyze lost-game replays in data/loser/*.json to find recurring failure
patterns the agent can fix.

For each replay we recover:
  * which seat makimakiver played (0/1)
  * opponent's deck composition (card-id histogram)
  * final win-condition (1 prizes / 2 deck-out / 3 no-active / 4 effect)
  * prize race timeline (prizes remaining per side per turn)
  * how many turns makimakiver actually attacked / evolved / attached energy
  * board state at the moment the game was decided
  * per-turn "did the active pokemon attack this turn" flag

The output is written to data/loser/_analysis.txt for inspection.
"""
import json
import os
from collections import Counter, defaultdict

LOSER_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "loser")
ME = "makimakiver"
REASON = {1: "all-prizes", 2: "deck-out", 3: "no-active", 4: "card-effect"}


def load_deck_names():
    """Map card-id -> human slug from data/cards.json for readable output."""
    p = os.path.join(os.path.dirname(__file__), "..", "data", "cards.json")
    try:
        with open(p) as f:
            cards = json.load(f)
        # cards.json may be a list of card dicts with 'cardId' / 'id'
        out = {}
        lst = cards if isinstance(cards, list) else list(cards.values())
        for c in lst:
            if isinstance(c, dict):
                cid = c.get("cardId") or c.get("id")
                name = c.get("name") or c.get("slug") or ""
                if cid is not None:
                    out[int(cid)] = name
        return out
    except Exception:
        return {}


NAMES = load_deck_names()


def name(cid):
    return NAMES.get(int(cid), str(cid))


def player_seat(step0_agents):
    """Return the seat index makimakiver occupies, given info.Agents."""
    for i, a in enumerate(step0_agents):
        if a.get("Name") == ME:
            return i
    return -1


def first_real_steps(steps, seat):
    """Yield (step_idx, elem) for steps where this seat had an ACTIVE select."""
    for i, pair in enumerate(steps):
        if seat >= len(pair):
            continue
        e = pair[seat]
        if e["status"] == "ACTIVE" and e["observation"].get("current"):
            yield i, e


def opponent_deck(steps, my_seat):
    """Recover opponent's 60-card deck by looking at the deck-submit step.

    The deck-submit step (step 1) has action == list of 60 card ids for the
    seat that submitted; the opponent's deck is the other seat's action.
    """
    opp = 1 - my_seat
    if len(steps) > 1 and len(steps[1]) > opp:
        act = steps[1][opp].get("action")
        if isinstance(act, list) and len(act) == 60:
            return act
    return []


def analyze_game(path):
    with open(path) as f:
        d = json.load(f)
    agents = d["info"]["Agents"]
    rewards = d.get("rewards", [None, None])
    steps = d["steps"]
    my_seat = player_seat(agents)
    if my_seat < 0:
        return None
    opp_seat = 1 - my_seat
    my_reward = rewards[my_seat] if my_seat < len(rewards) else None
    opp_name = agents[opp_seat].get("Name")

    # Find the RESULT log (type 23) and final current.
    reason = None
    result = None
    final_turn = None
    final_state = None
    # walk every step pair's logs
    for pair in steps:
        for seat_el in pair:
            obs = seat_el.get("observation") or {}
            for lg in obs.get("logs", []) or []:
                if lg.get("type") == 23:
                    reason = lg.get("reason")
                    result = lg.get("result")
            cur = obs.get("current")
            if cur is not None and cur.get("turn") is not None:
                final_state = cur
                final_turn = cur["turn"]

    # prize race: sample current.players[seat].prize length over turns
    prize_race = []  # (turn, my_prizes, opp_prizes)
    seen_turns = set()
    for i, e in first_real_steps(steps, my_seat):
        cur = e["observation"]["current"]
        t = cur.get("turn")
        if t in seen_turns:
            continue
        seen_turns.add(t)
        ps = cur.get("players", [])
        my_pr = len(ps[my_seat]["prize"]) if my_seat < len(ps) else None
        op_pr = len(ps[opp_seat]["prize"]) if opp_seat < len(ps) else None
        prize_race.append((t, my_pr, op_pr))

    # action-type histogram for makimakiver (what option types did we pick)
    opt_types = Counter()
    attacks = 0
    evolves = 0
    attaches = 0
    abilities = 0
    ends = 0
    plays = 0
    for i, e in first_real_steps(steps, my_seat):
        sel = e["observation"].get("select")
        if not sel:
            continue
        opts = sel.get("option", [])
        action = e.get("action", []) or []
        for idx in action:
            if 0 <= idx < len(opts):
                t = opts[idx].get("type")
                opt_types[t] += 1
                if t == 13:  # ATTACK
                    attacks += 1
                elif t == 9:  # EVOLVE
                    evolves += 1
                elif t == 8:  # ATTACH
                    attaches += 1
                elif t == 10:  # ABILITY
                    abilities += 1
                elif t == 14:  # END
                    ends += 1
                elif t == 7:  # PLAY
                    plays += 1

    # opponent deck histogram
    opp_deck = opponent_deck(steps, my_seat)
    opp_hist = Counter(opp_deck)

    # board state at loss: final_state players
    board = None
    if final_state:
        ps = final_state["players"]
        mine = ps[my_seat]
        theirs = ps[opp_seat]
        board = {
            "my_active": [name(c["id"]) for c in mine["active"] if c],
            "my_bench": [name(c["id"]) for c in mine["bench"]],
            "my_deck": mine.get("deckCount"),
            "my_hand": mine.get("handCount"),
            "my_prizes": len(mine.get("prize", [])),
            "opp_active": [name(c["id"]) for c in theirs["active"] if c],
            "opp_bench": [name(c["id"]) for c in theirs["bench"]],
            "opp_deck": theirs.get("deckCount"),
            "opp_prizes": len(theirs.get("prize", [])),
        }

    return {
        "file": os.path.basename(path),
        "opp_name": opp_name,
        "my_seat": my_seat,
        "my_reward": my_reward,
        "reason": REASON.get(reason, reason),
        "result": result,
        "final_turn": final_turn,
        "opt_types": dict(opt_types),
        "attacks": attacks,
        "evolves": evolves,
        "attaches": attaches,
        "abilities": abilities,
        "ends": ends,
        "plays": plays,
        "opp_deck_top": opp_hist.most_common(8),
        "prize_race": prize_race,
        "board": board,
    }


def main():
    files = sorted(
        os.path.join(LOSER_DIR, f)
        for f in os.listdir(LOSER_DIR)
        if f.endswith(".json")
    )
    out_lines = []
    games = []
    for p in files:
        try:
            g = analyze_game(p)
        except Exception as exc:
            out_lines.append(f"!! failed {p}: {exc}")
            continue
        if g is None:
            continue
        games.append(g)

    out_lines.append(f"### Analyzed {len(games)} lost games\n")
    # reason histogram
    reasons = Counter(g["reason"] for g in games)
    out_lines.append(f"Loss reasons: {dict(reasons)}")
    # opponent frequency
    opps = Counter(g["opp_name"] for g in games)
    out_lines.append(f"Opponents: {dict(opps)}")
    # final turn distribution
    turns = sorted(g["final_turn"] or 0 for g in games)
    out_lines.append(
        f"Final turn stats: min={turns[0] if turns else '-'} "
        f"median={turns[len(turns)//2] if turns else '-'} max={turns[-1] if turns else '-'}"
    )
    # avg actions
    n = max(1, len(games))
    out_lines.append(
        f"Avg per game: attacks={sum(g['attacks'] for g in games)/n:.1f} "
        f"evolves={sum(g['evolves'] for g in games)/n:.1f} "
        f"attaches={sum(g['attaches'] for g in games)/n:.1f} "
        f"abilities={sum(g['abilities'] for g in games)/n:.1f} "
        f"plays={sum(g['plays'] for g in games)/n:.1f} "
        f"ends={sum(g['ends'] for g in games)/n:.1f}"
    )

    # aggregate opponent decks
    opp_total = Counter()
    for g in games:
        for cid, c in g["opp_deck_top"]:
            opp_total[name(cid)] += c
    out_lines.append("\nMost common opponent cards across losses:")
    for nm, c in opp_total.most_common(20):
        out_lines.append(f"  {nm}: {c}")

    # per-game detail
    out_lines.append("\n" + "=" * 70 + "\nPER-GAME DETAIL\n" + "=" * 70)
    for g in games:
        out_lines.append("")
        out_lines.append(
            f"## {g['file']}  vs {g['opp_name']}  seat={g['my_seat']}  "
            f"reward={g['my_reward']}  reason={g['reason']}  final_turn={g['final_turn']}"
        )
        out_lines.append(f"  actions: attacks={g['attacks']} evolves={g['evolves']} "
                         f"attaches={g['attaches']} abilities={g['abilities']} "
                         f"plays={g['plays']} ends={g['ends']}")
        out_lines.append(f"  opp deck top: " + ", ".join(
            f"{name(cid)}x{c}" for cid, c in g["opp_deck_top"]))
        if g["board"]:
            b = g["board"]
            out_lines.append(
                f"  FINAL BOARD: my_active={b['my_active']} "
                f"my_bench={b['my_bench']} my_prizes={b['my_prizes']} "
                f"my_deck={b['my_deck']} my_hand={b['my_hand']}"
            )
            out_lines.append(
                f"                opp_active={b['opp_active']} "
                f"opp_bench({len(b['opp_bench'])}) opp_prizes={b['opp_prizes']} "
                f"opp_deck={b['opp_deck']}"
            )
        # prize race last few
        if g["prize_race"]:
            tail = g["prize_race"][-6:]
            out_lines.append(
                "  prize race (turn,my,opp): " + " ".join(
                    f"({t},{mp},{op})" for t, mp, op in tail))

    txt = "\n".join(out_lines)
    out_path = os.path.join(LOSER_DIR, "_analysis.txt")
    with open(out_path, "w") as f:
        f.write(txt)
    print(txt)
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
