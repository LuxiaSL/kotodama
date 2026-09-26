---
name: warmup
description: Ground a fresh kotodama session fast — reads the live handoff, canon index, and active program docs in the right order. Use at session start or after /compact.
---

# /warmup — kotodama session grounding

Read these in order, skipping any that are already in context:

1. `research/planning/INDEX.md` — canon index; note which docs are LIVE.
2. Any `research/planning/HANDOFF-*.md` whose STATUS header says LIVE
   (as of 2026-07-04: `HANDOFF-7B-THROUGHPUT-2026-07-04.md`). A LIVE handoff
   is the session-reload entry point: repo state, operational laws, next
   actions with reasoning. Trust it over memory of prior conversations.
3. The SPEC the handoff points to (e.g. `SPEC-7B-THROUGHPUT-2026-07-03.md`)
   — results tables and ranked plans live THERE, not in the handoff.
4. If the work touches code: read the actual source files the plan names
   before assuming anything (API signatures drift; source is truth).

Then confirm operational state before touching the cluster:

- `forticlient vpn status` — reconnect with
  `forticlient vpn connect borgs-nodes` if node1 is unreachable (it drops
  most nights; this is ALWAYS the first move, not NIC diagnosis).
- `ssh node1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
  and `curl -s 127.0.0.1:7000/api/v1/jobs` (via ssh) — GPU + Heimdall queue
  state. Never assume the GPUs are free; other tenants and the user's own
  jobs share node1.

Finish by telling the user: current program state in one breath, what the
next planned action is, and anything surprising found during recon.
