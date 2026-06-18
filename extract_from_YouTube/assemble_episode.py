"""Assemble the final episode from hand-extracted SemanticEvents.

The Anthropic extraction step (extract.py) needs ANTHROPIC_API_KEY, absent in
this environment. These events were extracted by reading the whisper transcript
(/tmp/ptcg_transcript.json) and applying prompts.py's schema. They run through
the REAL schema.py validators + Episode builder, so match.json is the tool's
genuine output format.

Players (index order from the CLI convention):
  0 = Jack Pitcher
  1 = Andrew Hedrick   <- match winner (2-0)

ASR note: caster audio garbles card names (Munkidori="monkey dory",
Drakloak="trick/jerk cloak", Dreepy="dreepee/3p", Dunsparce="dunce bars",
Budew="bidu", Crispin="crispin", Fezandipiti="pheasantipity"). Names below are
normalized; combat attribution in the messy mid-game is honestly low-confidence.
"""
import json
from schema import SemanticEvent, Episode

URL = "https://www.youtube.com/watch?v=m8np08cT-TQ"
PLAYERS = ["Jack Pitcher", "Andrew Hedrick"]
WINNER = 1  # Andrew Hedrick took the series 2-0

E = []  # raw event dicts; validated by SemanticEvent below

def ev(**k):
    E.append(k)

# ===================== GAME 1 (Andrew wins; Jack concedes) =================
ev(turn=0, player=0, type="note", t_start=186.0, source_text="does seem like Jack has chosen to go first", confidence=0.85)
ev(turn=0, player=0, type="setup", card="Dunsparce", to_zone="active", t_start=421.9, source_text="will be a Dunspar start", confidence=0.8)
ev(turn=0, player=1, type="setup", card="Munkidori", to_zone="active", t_start=421.9, source_text="up against the monkey dory of Andrew", confidence=0.8)

ev(turn=1, player=0, type="play_item", card="Ultra Ball", t_start=421.9, source_text="an opening ultra ball for Jack", confidence=0.85)
ev(turn=1, player=0, type="discard", card="Lillie's Determination", from_zone="hand", to_zone="discard", t_start=426.4, source_text="The discord of the lilies determination gives a lot of information", confidence=0.55)
ev(turn=1, player=0, type="play_basic", card="Dreepy", to_zone="bench", t_start=459.9, source_text="dreepee will hit the field", confidence=0.8)
ev(turn=1, player=0, type="attach_energy", target="Dreepy", t_start=465.8, source_text="does he put an energy on it? He surely does", confidence=0.8)
ev(turn=1, player=0, type="end_turn", t_start=465.8, source_text="now a pass to Andrew Hedrick", confidence=0.9)

ev(turn=2, player=1, type="play_stadium", card="Risky Ruins", to_zone="stadium", t_start=537.2, source_text="the placement of the risky ruins before the buddy buddy poppin", confidence=0.6)
ev(turn=2, player=1, type="play_supporter", card="Crispin", t_start=826.6, source_text="chip he found lily's determination ... chrispin finds the energy", confidence=0.55)
ev(turn=2, player=1, type="attach_energy", energy_type="darkness", target="Munkidori", t_start=546.9, source_text="Crispin putting a darkness energy on monkey dory", confidence=0.55)
ev(turn=2, player=1, type="attack", card="Munkidori", attack_name="Mind Bend", target="Dunsparce", amount=70, t_start=603.0, source_text="This is going to KO that done", confidence=0.5)
ev(turn=2, player=1, type="knockout", target="Dunsparce", t_start=605.0, source_text="get Andrew ahead in the prize trade", confidence=0.6)
ev(turn=2, player=1, type="take_prize", player_note=None, amount=1, from_zone="prize", t_start=605.0, source_text="get Andrew ahead in the prize trade", confidence=0.6)

ev(turn=3, player=0, type="end_turn", t_start=807.4, source_text="just to pass now from jack", confidence=0.6)

ev(turn=4, player=1, type="ability", card="Munkidori", t_start=874.0, source_text="Adrena brain bouncing those two damage counters over to the jerk cloak", confidence=0.55)
ev(turn=4, player=1, type="attack", card="Drakloak", attack_name="Dragon Headbutt", target="Drakloak", amount=90, t_start=880.9, source_text="dragon headbutt 70 damage plus the 20 that's 90", confidence=0.5)
ev(turn=4, player=1, type="knockout", target="Drakloak", t_start=888.5, source_text="that is a knockout", confidence=0.6)
ev(turn=4, player=1, type="take_prize", amount=2, from_zone="prize", t_start=890.0, source_text="andrew up a quick two prize cards", confidence=0.6)

ev(turn=5, player=0, type="play_basic", card="Dreepy", to_zone="active", t_start=897.7, source_text="Jack pushing the dreepee", confidence=0.6)
ev(turn=5, player=0, type="game_result", winner=1, t_start=920.0, source_text="and he's just going to concede", confidence=0.9)

# ===================== GAME 2 (Andrew wins, takes the title) ===============
ev(turn=6, player=0, type="note", t_start=967.7, source_text="so jack will be choosing to go first (game two)", confidence=0.85)
ev(turn=6, player=0, type="setup", card="Dreepy", to_zone="active", t_start=1037.6, source_text="there is a dreepee", confidence=0.75)
ev(turn=6, player=1, type="setup", card="Budew", to_zone="active", t_start=1067.2, source_text="andrew's itchy pollen ... that pokemon being the starter", confidence=0.7)

ev(turn=7, player=0, type="play_item", card="Poke Pad", t_start=1093.9, source_text="going to grab joe cloak here from the poke pad", confidence=0.65)
ev(turn=7, player=0, type="search", card="Drakloak", from_zone="deck", to_zone="hand", t_start=1093.9, source_text="grab joe cloak ... get himself some draw on the next turn", confidence=0.6)
ev(turn=7, player=0, type="end_turn", t_start=1108.5, source_text="no additional pokemon played down", confidence=0.7)

ev(turn=8, player=1, type="play_item", card="Buddy Buddy Poffin", t_start=1122.0, source_text="buddy buddy pauffin and lily's determination", confidence=0.8)
ev(turn=8, player=1, type="play_basic", card="Dreepy", to_zone="bench", amount=3, t_start=1225.0, source_text="three drippy an energy already attached", confidence=0.7)
ev(turn=8, player=1, type="attach_energy", target="Dreepy", t_start=1213.0, source_text="the fire for the drippy", confidence=0.6)
ev(turn=8, player=1, type="play_supporter", card="Lillie's Determination", amount=8, t_start=1225.0, source_text="a hand shuffled into the deck and still eight more cards to work with", confidence=0.75)
ev(turn=8, player=1, type="ability", card="Budew", attack_name="Itchy Pollen", t_start=1106.0, source_text="andrew's itchy pollen ... jack unable to play item cards", confidence=0.7)

ev(turn=9, player=0, type="ability", card="Drakloak", attack_name="Recon Directive", amount=2, t_start=1242.2, source_text="he'll at most see two cards with a recon directive", confidence=0.65)
ev(turn=9, player=0, type="play_basic", card="Munkidori", to_zone="bench", t_start=1282.5, source_text="just debating the bench of monkey dory looks like it will come down", confidence=0.6)
ev(turn=9, player=0, type="play_supporter", card="Lillie's Determination", t_start=1286.6, source_text="Lily's determination does hold off on using risky ruins", confidence=0.6)

ev(turn=10, player=1, type="play_supporter", card="Judge", amount=4, t_start=1373.8, source_text="go with the judge having each player shuffle their hand into the deck and get four cards", confidence=0.75)
ev(turn=10, player=1, type="attach_energy", energy_type="darkness", target="Munkidori", t_start=1363.3, source_text="Andrew will get a darkness energy established on to monkey dory", confidence=0.6)
ev(turn=10, player=1, type="ability", card="Budew", attack_name="Itchy Pollen", t_start=1479.2, source_text="Itchy pollen once again jack unable to play item cards", confidence=0.65)

ev(turn=11, player=0, type="play_supporter", card="Lillie's Determination", amount=8, t_start=1518.2, source_text="eight cards off of lilies", confidence=0.65)
ev(turn=11, player=0, type="ability", card="Drakloak", attack_name="Recon Directive", t_start=1538.1, source_text="Recon directive does find him a basic dark", confidence=0.6)
ev(turn=11, player=0, type="attach_energy", energy_type="darkness", target="Munkidori", t_start=1542.9, source_text="find him a basic dark ... immediate answer to this third itchy pollen", confidence=0.55)

ev(turn=12, player=1, type="play_basic", card="Munkidori", to_zone="bench", t_start=1685.5, source_text="he's got second monkey Dory", confidence=0.6)
ev(turn=12, player=1, type="evolve", card="Dragapult ex", target="Drakloak", t_start=1682.4, source_text="we do see the dragapult EX evolving", confidence=0.7)
ev(turn=12, player=1, type="attach_energy", energy_type="darkness", target="Munkidori", t_start=1733.9, source_text="there's that attachment of the basic dark", confidence=0.55)
ev(turn=12, player=1, type="retreat", card="Budew", t_start=1745.9, source_text="retreat into dragapult EX", confidence=0.55)
ev(turn=12, player=1, type="attack", card="Dragapult ex", attack_name="Jet Headbutt", amount=70, t_start=1751.3, source_text="the jet headbutt here avoiding putting damage on jack side", confidence=0.55)

ev(turn=13, player=0, type="play_item", card="Poke Pad", t_start=1791.8, source_text="Andrew has unlocked all of jack's items ... pokepad finding a replacement done sparse", confidence=0.6)
ev(turn=13, player=0, type="play_basic", card="Dunsparce", to_zone="bench", t_start=1797.6, source_text="finding a replacement done sparse getting that into play", confidence=0.6)
ev(turn=13, player=0, type="play_supporter", card="Unfair Stamp", t_start=1805.4, source_text="he does have his ace spec unfair stamp waiting in the hand", confidence=0.55)

ev(turn=14, player=1, type="ability", card="Dragapult ex", attack_name="Phantom Dive", amount=200, t_start=1959.9, source_text="that'll be 200 damage onto his active dragon poulton 30 onto each of the dracloch", confidence=0.55)

# pace-of-play penalty -> Jack needs two fewer prizes (a real swing in the game)
ev(turn=15, player=1, type="note", t_start=2348.7, source_text="extension issued ... jack now needing to take two less prize cards (pace-of-play penalty on Andrew)", confidence=0.7)

ev(turn=16, player=0, type="play_supporter", card="Crispin", t_start=2705.0, source_text="Crispin is found", confidence=0.5)
ev(turn=16, player=0, type="ability", card="Munkidori", attack_name="Adrena-Brain", amount=20, t_start=2531.9, source_text="with the monkey dories you push 20 damage counters over twice", confidence=0.5)
ev(turn=16, player=0, type="attack", card="Munkidori", attack_name="Mind Bend", amount=60, target="Drakloak", t_start=2531.9, source_text="three prize card turn now needing to just take one prize card to win", confidence=0.5)
ev(turn=16, player=0, type="take_prize", amount=3, from_zone="prize", t_start=2515.4, source_text="a three prize card turn now needing to just take one prize card", confidence=0.6)

ev(turn=17, player=1, type="play_item", card="Crushing Hammer", t_start=2557.6, source_text="Does find a crushing hammer which could remove a darkness energy from monkey dory", confidence=0.7)
ev(turn=17, player=1, type="coin_flip", coin="heads", t_start=2565.8, source_text="There's the flip ... it's a heads", confidence=0.8)
ev(turn=17, player=1, type="discard", card="Darkness Energy", target="Munkidori", from_zone="attached", to_zone="discard", t_start=2580.9, source_text="you just have to remove this dark energy and he is going to go for it", confidence=0.65)
ev(turn=17, player=1, type="play_item", card="Crushing Hammer", t_start=2695.3, source_text="crushing hammer this is another huge flippin", confidence=0.7)
ev(turn=17, player=1, type="coin_flip", coin="heads", t_start=2698.6, source_text="It is another heads and that is going to remove the fire energy", confidence=0.8)
ev(turn=17, player=1, type="discard", card="Fire Energy", from_zone="attached", to_zone="discard", t_start=2698.6, source_text="that is going to remove the fire energy", confidence=0.65)
ev(turn=17, player=1, type="play_supporter", card="Crispin", t_start=2728.7, source_text="out comes chrisvin finding two basic energy of differing types one fire one psychic", confidence=0.65)
ev(turn=17, player=1, type="evolve", card="Dragapult ex", target="Drakloak", t_start=2741.9, source_text="down will go this dragapult EX", confidence=0.65)
ev(turn=17, player=1, type="ability", card="Dragapult ex", attack_name="Phantom Dive", amount=20, t_start=2746.9, source_text="20 damage bounced over from andrew side", confidence=0.55)
ev(turn=17, player=1, type="knockout", target="Dunsparce", t_start=2825.0, source_text="down goes dunce bars down to just one prize card", confidence=0.6)
ev(turn=17, player=1, type="take_prize", amount=1, from_zone="prize", t_start=2790.9, source_text="down to just one prize card is andrew headrick", confidence=0.6)

ev(turn=18, player=0, type="play_supporter", card="Crispin", t_start=2905.7, source_text="two different energies to grab from the crisp in", confidence=0.5)
ev(turn=18, player=0, type="attach_energy", target="Munkidori", t_start=2911.1, source_text="He will attach to this active monkey dory", confidence=0.55)
ev(turn=18, player=0, type="attack", card="Munkidori", attack_name="Mind Bend", amount=60, t_start=2927.7, source_text="he is going for it mind bend that's 60 damage", confidence=0.55)

ev(turn=19, player=1, type="ability", card="Munkidori", attack_name="Adrena-Brain", t_start=2930.7, source_text="adrenaline brain available will bounce them both over to the active", confidence=0.6)
ev(turn=19, player=1, type="retreat", card="Dragapult ex", t_start=2939.3, source_text="retreat his dragapult into monkey dory", confidence=0.6)
ev(turn=19, player=1, type="attack", card="Munkidori", attack_name="Mind Bend", amount=60, t_start=2939.6, source_text="mind bend takes the KO", confidence=0.6)
ev(turn=19, player=1, type="knockout", t_start=2939.6, source_text="takes the KO", confidence=0.7)
ev(turn=19, player=1, type="take_prize", amount=1, from_zone="prize", t_start=2943.2, source_text="andrew headrick a five-time regional champion", confidence=0.7)
ev(turn=19, player=1, type="game_result", winner=1, t_start=2943.2, source_text="andrew headrick a five-time regional champion (wins match 2-0)", confidence=0.95)

# ----- validate via the real schema, drop fields it doesn't know -----
KNOWN = set(SemanticEvent.model_fields.keys())
events = []
for d in E:
    clean = {k: v for k, v in d.items() if k in KNOWN}
    events.append(SemanticEvent(**clean))

episode = Episode(players=PLAYERS, events=events, winner=WINNER,
                  source_url=URL, emit_cabt=True).build()

with open("match.json", "w") as f:
    json.dump(episode, f, indent=2, ensure_ascii=False)

print(f"events: {len(events)}")
print(f"steps:  {len(episode['steps'])}")
print(f"rewards:{episode['rewards']}  (winner=Andrew Hedrick)")
print("wrote match.json")
