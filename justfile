set dotenv-load := true
set shell := ["bash", "-cu"]

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PROJECT := "strategy-lab"
PORT := env_var_or_default("PI_COMS_NET_PORT", "52965")
HOST := env_var_or_default("PI_COMS_NET_HOST", "127.0.0.1")

# Repo holding the coms extensions (coms.ts / coms-net.ts / minimal.ts / theme-cycler.ts)
PI_REPO := "/Users/makimakiver/pi-vs-claude-code"

# Default: show available commands
default:
    @just --list

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

# Install project dependencies
install:
    bun install

# Check required tools
check:
    @command -v bun >/dev/null || (echo "❌ bun not found" && exit 1)
    @command -v pi >/dev/null || (echo "❌ pi not found" && exit 1)
    @command -v just >/dev/null || (echo "❌ just not found" && exit 1)
    @echo "✅ bun, pi, just found"

# ─────────────────────────────────────────────
# Coms-net server
# ─────────────────────────────────────────────

# Start local agent communication hub
server:
    @echo "🚀 Starting local coms-net hub on {{HOST}}:{{PORT}}"
    @lsof -ti :{{PORT}} | xargs -r kill -TERM 2>/dev/null || true
    PI_COMS_NET_HOST={{HOST}} \
    PI_COMS_NET_PORT={{PORT}} \
    PI_COMS_NET_PROJECT={{PROJECT}} \
    bun scripts/coms-net-server.ts

# Start LAN-visible hub; requires PI_COMS_NET_AUTH_TOKEN
server-lan:
    @test -n "$${PI_COMS_NET_AUTH_TOKEN:-}" || (echo "❌ Set PI_COMS_NET_AUTH_TOKEN first" && exit 1)
    @echo "🌐 Starting LAN coms-net hub on 0.0.0.0:{{PORT}}"
    @lsof -ti :{{PORT}} | xargs -r kill -TERM 2>/dev/null || true
    PI_COMS_NET_HOST=0.0.0.0 \
    PI_COMS_NET_PORT={{PORT}} \
    PI_COMS_NET_PROJECT={{PROJECT}} \
    bun scripts/coms-net-server.ts

# Kill local hub
kill-server:
    @lsof -ti :{{PORT}} | xargs -r kill -TERM 2>/dev/null || true
    @echo "🛑 killed anything on port {{PORT}}"

# ─────────────────────────────────────────────
# Single Pi agents
# ─────────────────────────────────────────────

# Base coms-net Pi agent
agent name purpose model="gpt-5.5":
    pi \
      -e extensions/coms-net.ts \
      -e extensions/minimal.ts \
      -e extensions/theme-cycler.ts \
      --provider openai \
      --model "{{model}}" \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{purpose}}" \
      --project "{{PROJECT}}"

# Strategy planner agent
planner:
    just agent planner "high-level strategy planner; decomposes game plans into testable hypotheses" "gpt-5.5"

# Strategy inventor agent
strategist:
    just agent strategist "creates new candidate strategies, heuristics, and policy rules" "gpt-5.5"

# Critic / verifier agent
critic:
    just agent critic "checks strategic claims against rules, math, legality, and counterplay" "gpt-5.5"

# Simulation / experiment agent
sim:
    just agent sim "runs experiments, compares win rates, and reports empirical evidence" "gpt-5.5"

# Code builder agent
builder:
    just agent builder "turns selected strategies into executable policy code" "gpt-5.5"

# Claude-backed agent, if your Pi provider config supports it
claude name purpose:
    pi \
      -e extensions/coms-net.ts \
      -e extensions/minimal.ts \
      -e extensions/theme-cycler.ts \
      --model claude-opus-4-7 \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{purpose}}" \
      --project "{{PROJECT}}"

# ─────────────────────────────────────────────
# Open agents in macOS Terminal
# ─────────────────────────────────────────────

# Open a recipe in a new macOS Terminal window
open recipe:
    osascript -e 'tell application "Terminal" to do script "cd {{justfile_directory()}} && just {{recipe}}"'

# Open full strategy team
team:
    just open planner
    just open strategist
    just open critic
    just open sim
    just open builder

# Start server + team manually in separate terminals
all:
    just open server
    sleep 1
    just team

# ─────────────────────────────────────────────
# Local same-machine coms, no hub
# ─────────────────────────────────────────────

# Raw same-machine coms agent; pass pi flags through, e.g.:
#   just local-coms --name strategist --cname strategist --purpose "..." --color "#FF7EDB"
local-coms *args:
    pi \
      -e {{PI_REPO}}/extensions/coms.ts \
      -e {{PI_REPO}}/extensions/minimal.ts \
      -e {{PI_REPO}}/extensions/theme-cycler.ts \
      {{args}}

# Local socket-based Pi agent
local name purpose:
    pi \
      -e extensions/coms.ts \
      -e extensions/minimal.ts \
      -e extensions/theme-cycler.ts \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{purpose}}" \
      --project "{{PROJECT}}"

# Local coms agent with a custom color (quotes each value, so '#hex' is safe)
local-color name purpose color:
    pi \
      -e {{PI_REPO}}/extensions/coms.ts \
      -e {{PI_REPO}}/extensions/minimal.ts \
      -e {{PI_REPO}}/extensions/theme-cycler.ts \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{purpose}}" \
      --color "{{color}}" \
      --project "{{PROJECT}}"

# Protocol-aware coms agent wired into the discussion<->Claude loop.
# Reads discussion/PROTOCOL.md and uses discussion/discussion.log as the channel.
loop-agent name role:
    pi \
      -e {{PI_REPO}}/extensions/coms.ts \
      -e {{PI_REPO}}/extensions/minimal.ts \
      -e {{PI_REPO}}/extensions/theme-cycler.ts \
      --name "{{name}}" \
      --cname "{{name}}" \
      --purpose "{{role}}. At session start, read ./discussion/PROTOCOL.md and follow it exactly. Append every turn to ./discussion/discussion.log. When the team agrees on a testable hypothesis, write an '@claude EXPERIMENT:' block naming candidate and baseline agent modules, then read discussion.log until '@discussion RESULT:' or '@discussion ERROR:' appears and continue." \
      --project "{{PROJECT}}"

# Open the full protocol-aware team in macOS Terminal windows.
# Note: no 'sim' agent — the experiment/sim role is handled by Claude Code's /loop.
loop-team:
    just open "loop-agent planner 'high-level strategy planner; decomposes game plans into testable hypotheses'"
    just open "loop-agent strategist 'creates new candidate strategies, heuristics, and policy rules'"
    just open "loop-agent critic 'checks strategic claims against rules, math, legality, and counterplay'"

local-planner:
    just local planner "local planning agent"

local-strategist:
    just local strategist "local strategy creation agent"

local-critic:
    just local critic "local verification agent"

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

# Clean Pi communication state
clean-coms:
    rm -rf ~/.pi/coms/projects/{{PROJECT}}
    rm -rf ~/.pi/coms-net
    @echo "🧹 cleaned coms state for {{PROJECT}}"

# Show local server files
status:
    @echo "Project: {{PROJECT}}"
    @echo "Port:    {{PORT}}"
    @echo "Host:    {{HOST}}"
    @echo ""
    @echo "~/.pi/coms-net:"
    @ls -la ~/.pi/coms-net 2>/dev/null || true
    @echo ""
    @echo "Port usage:"
    @lsof -i :{{PORT}} || true
