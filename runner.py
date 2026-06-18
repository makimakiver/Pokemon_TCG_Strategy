"""Local self-play match runner for the cabt engine.

Pits two agent modules against each other over N games and reports win rates.
Each agent module must expose `agent(obs_dict) -> list[int]` and `my_deck`.
Agents keep per-module global state (plan/pre_turn), so the two sides must be
DIFFERENT import names to avoid sharing globals.
"""
import argparse
import importlib
import sys
import traceback

from cg.game import battle_start, battle_select, battle_finish


def deck_request(mod):
    """Get an agent's 60-card deck by calling it with a null-select observation."""
    try:
        d = mod.agent({"select": None, "logs": [], "current": None})
        if isinstance(d, list) and len(d) == 60:
            return d
    except Exception:
        pass
    return list(mod.my_deck)


REASON = {1: "all-prizes", 2: "deck-out", 3: "no-active", 4: "card-effect"}


def play_one(mod0, mod1, max_steps=10000, reasons=None, loss_detail=None):
    """Play a single game. Returns winner index (0/1) or -1 for draw/error."""
    deck0, deck1 = deck_request(mod0), deck_request(mod1)
    obs, start = battle_start(deck0, deck1)
    if obs is None:
        print(f"  battle_start failed: errorPlayer={start.errorPlayer} errorType={start.errorType}")
        return -1
    mods = (mod0, mod1)
    last_reason = None
    try:
        for _ in range(max_steps):
            cur = obs.get("current")
            if cur is None:
                break
            # Capture the match-result reason from the logs if present.
            for lg in (obs.get("logs") or []):
                if lg.get("type") == 23 and lg.get("reason") is not None:
                    last_reason = lg["reason"]
            if cur.get("result", -1) != -1:
                if reasons is not None:
                    reasons[REASON.get(last_reason, last_reason)] += 1
                    reasons["_last"] = REASON.get(last_reason, last_reason)
                if loss_detail is not None:
                    loser = 1 - cur["result"]
                    lp = cur["players"][loser]
                    n_poke = (1 if lp["active"] and lp["active"][0] else 0) + len(lp["bench"])
                    loss_detail.append({
                        "loser": loser,
                        "turn": cur.get("turn"),
                        "loser_pokemon_on_board": n_poke,
                        "loser_deck": lp.get("deckCount"),
                        "loser_hand": lp.get("handCount"),
                        "reason": REASON.get(last_reason, last_reason),
                    })
                return cur["result"]
            if obs.get("select") is None:
                break
            who = cur["yourIndex"]
            try:
                action = mods[who].agent(obs)
            except Exception:
                print(f"  agent {who} crashed:")
                traceback.print_exc()
                return 1 - who  # crashing agent forfeits
            if not isinstance(action, list):
                action = list(action)
            obs = battle_select(action)
        # Fell out of loop without a result.
        cur = obs.get("current") or {}
        return cur.get("result", -1)
    finally:
        battle_finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="agents.main", help="agent module for side A")
    ap.add_argument("--b", default="agents.main_megalucario_backup", help="agent module for side B")
    ap.add_argument("-n", "--games", type=int, default=20)
    args = ap.parse_args()

    modA = importlib.import_module(args.a)
    modB = importlib.import_module(args.b)
    print(f"A = {args.a}   vs   B = {args.b}   ({args.games} games, sides swapped each game)\n")

    from collections import Counter
    wins = {"A": 0, "B": 0, "draw": 0}
    reasons = Counter()  # how games ended (win condition that closed the game)
    by_winner = Counter()  # (winner, reason)
    a_losses = []  # board states when A lost
    for g in range(args.games):
        reasons["_last"] = None
        ld = []
        # Swap who goes first/which seat to remove seat bias. A's seat each game:
        a_seat = 0 if g % 2 == 0 else 1
        if g % 2 == 0:
            res = play_one(modA, modB, reasons=reasons, loss_detail=ld)   # A is player0
            outcome = "A" if res == 0 else "B" if res == 1 else "draw"
        else:
            res = play_one(modB, modA, reasons=reasons, loss_detail=ld)   # A is player1
            outcome = "A" if res == 1 else "B" if res == 0 else "draw"
        if outcome == "B" and ld and ld[0]["loser"] == a_seat:
            a_losses.append(ld[0])
        by_winner[f"{outcome}:{reasons.get('_last')}"] += 1
        wins[outcome] += 1
        print(f"  game {g+1:3d}: winner = {outcome:4s}   "
              f"(A {wins['A']} / B {wins['B']} / draw {wins['draw']})")
        sys.stdout.flush()

    total = args.games
    print(f"\n==== RESULT over {total} games ====")
    print(f"  A ({args.a}): {wins['A']}  ({100*wins['A']/total:.1f}%)")
    print(f"  B ({args.b}): {wins['B']}  ({100*wins['B']/total:.1f}%)")
    print(f"  draws: {wins['draw']}")
    del reasons["_last"]
    print(f"  win conditions: {dict(reasons)}")
    print(f"  winner:reason  : {dict(by_winner)}")
    if a_losses:
        print(f"  --- A's {len(a_losses)} losses (board state at loss) ---")
        for d in a_losses:
            print(f"    turn={d['turn']} reason={d['reason']} "
                  f"A_pokemon_on_board={d['loser_pokemon_on_board']} "
                  f"A_deck={d['loser_deck']} A_hand={d['loser_hand']}")


if __name__ == "__main__":
    main()
