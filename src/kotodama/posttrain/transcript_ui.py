"""A small web UI for driving a base-mode server with plain-text transcripts (no special tokens).

One person writes one side of a ``Name: text`` transcript; the model writes the other (or both). Each model turn
comes back as N candidates, each already cut at the next speaker exactly as self-play cuts it and tagged with the
mechanical vetoes it would trip (kotodama.posttrain.selfplay.evaluate) — so a session is also a live look at the
flywheel's filter. Click a candidate to commit it; reroll; let the model write your turn; edit or delete turns;
export the transcript as a training row.

    python -m kotodama.posttrain.transcript_ui --backend masked=http://HOST:2352 --backend base=http://HOST:2353 \\
        [--port 3344]

Then open http://localhost:3344. Prompts are rendered here (kotodama.posttrain.transcript), so the page only ever
sends turns, never prompt text. Standard library only, plus httpx for the backend calls.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import sys
from typing import Any, Optional

from kotodama.posttrain.selfplay import SelfPlayConfig, evaluate, split_completion
from kotodama.posttrain.transcript import Transcript, Turn, render_prompt, stop_sequences

logger = logging.getLogger(__name__)

MAX_N = 8


def backend_info(client: Any, url: str) -> dict[str, Any]:
    """The server's /info name and checkpoint (what is actually listening there)."""
    try:
        r = client.get(url.rstrip("/") + "/info", timeout=10)
        r.raise_for_status()
        d = r.json()
        return {"name": d.get("name", "?"), "checkpoint": d.get("checkpoint", "?"), "ok": True}
    except Exception as e:  # an unreachable backend is shown, not fatal
        return {"name": f"unreachable ({type(e).__name__})", "checkpoint": "", "ok": False}


def candidates(client: Any, url: str, req: dict[str, Any]) -> list[dict[str, Any]]:
    """N candidate turns for ``req["speaker"]`` after ``req["turns"]``, cut and vetted as self-play does."""
    cast = [c for c in req.get("cast", []) if isinstance(c, str) and c.strip()]
    speaker = req["speaker"]
    if speaker not in cast:
        cast.append(speaker)
    turns = [Turn(t["speaker"], t["text"]) for t in req.get("turns", [])]
    header = str(req.get("header", "")).strip()
    n = max(1, min(int(req.get("n", 4)), MAX_N))
    max_tokens = max(8, min(int(req.get("max_tokens", 256)), 1024))
    stops = stop_sequences(cast)
    body: dict[str, Any] = {
        "prompt": render_prompt(Transcript(tuple(turns), header), speaker), "n": n, "max_tokens": max_tokens,
        "temperature": float(req.get("temperature", 0.75)), "top_p": 0.0,
        "frequency_penalty": max(0.0, float(req.get("repetition_penalty", 1.2)) - 1.0),
        "stop": stops, "return_token_ids": True,
    }
    steer = [s for s in req.get("steer", []) if isinstance(s, dict) and s.get("name") and s.get("alpha")]
    if steer:
        body["vectors"] = [{"name": s["name"], "alpha": float(s["alpha"])} for s in steer]
    r = client.post(url.rstrip("/") + "/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    cfg = SelfPlayConfig(max_tokens=max_tokens, min_turns=int(req.get("min_turns", 4)))
    out = []
    for ch in r.json()["choices"]:
        ids = (ch.get("token_ids") or {}).get("completion") or []
        text, ended_by = split_completion(ch["text"], stops, ch.get("finish_reason", "stop"), len(ids) >= max_tokens)
        c = evaluate(text, ended_by, speaker, turns, cast, header, cfg)
        out.append({"text": c.text, "ended_by": c.ended_by, "vetoes": c.vetoes, "cut_label": c.cut_label,
                    "selfcopy": round(c.selfcopy_frac, 3)})
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcript Loom</title>
<style>
:root{--bg:#f7f6f3;--panel:#fff;--ink:#1d1d1f;--dim:#6b6b70;--line:#e3e1dc;--accent:#5b4bc4;--you:#eef0ff;--warn:#b4361f;--ok:#2f7d4f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;--panel:#1b1b1f;--ink:#e8e8ea;--dim:#9a9aa2;--line:#2c2c32;--accent:#a99bff;--you:#1f2133;--warn:#ff8a70;--ok:#7fd3a0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,sans-serif}
.wrap{display:grid;grid-template-columns:280px 1fr;gap:16px;padding:16px;max-width:1400px;margin:auto}
@media (max-width:820px){.wrap{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
label{display:block;font-size:12px;color:var(--dim);margin:10px 0 4px}
input,select,textarea{width:100%;padding:7px 8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit}
textarea{resize:vertical}.row{display:flex;gap:8px}.row>*{flex:1}
button{padding:7px 12px;border-radius:7px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer;font:inherit}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}button:disabled{opacity:.5;cursor:wait}
.turn{padding:8px 10px;border-radius:8px;margin:6px 0;white-space:pre-wrap;position:relative}
.turn.you{background:var(--you)}.turn .who{font-weight:600;margin-right:6px}
.turn .tools{position:absolute;right:6px;top:4px;display:none;gap:4px}.turn:hover .tools{display:flex}
.turn .tools button{padding:1px 6px;font-size:12px}
.cand{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0;cursor:pointer;white-space:pre-wrap}
.cand:hover{border-color:var(--accent)}.tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:9px;margin-right:4px;border:1px solid var(--line);color:var(--dim)}
.tag.bad{color:var(--warn);border-color:var(--warn)}.tag.good{color:var(--ok);border-color:var(--ok)}
.header{color:var(--dim);font-style:italic;white-space:pre-wrap;margin-bottom:8px}.status{color:var(--dim);font-size:12px;margin-top:8px}
h1{font-size:16px;margin:0 0 4px}.small{font-size:12px;color:var(--dim)}
</style></head><body>
<div class="wrap">
<div class="panel">
  <h1>Transcript Loom</h1><div class="small">plain-text transcripts · no special tokens</div>
  <label>Backend</label><select id="backend"></select><div class="small" id="binfo"></div>
  <label>Header (setting, participants)</label><textarea id="header" rows="3" placeholder="A conversation between ..."></textarea>
  <div class="row"><div><label>You</label><input id="you" value="User"></div><div><label>Model</label><input id="them" value="Model"></div></div>
  <div class="row"><div><label>Temperature</label><input id="temp" type="number" step="0.05" value="0.75"></div>
  <div><label>Repetition</label><input id="rep" type="number" step="0.05" value="1.2"></div></div>
  <div class="row"><div><label>Candidates</label><input id="n" type="number" min="1" max="8" value="4"></div>
  <div><label>Max tokens</label><input id="maxtok" type="number" value="256"></div></div>
  <div class="row"><div><label>vrep α (anti-loop: −)</label><input id="vrep" type="number" step="0.01" value="0"></div>
  <div><label>V7 α (entropy: +)</label><input id="v7" type="number" step="0.01" value="0"></div></div>
  <div class="row" style="margin-top:12px"><button onclick="exportJson()">Export</button><button onclick="resetAll()">Reset</button></div>
  <div class="status" id="status">ready</div>
</div>
<div class="panel">
  <div class="header" id="hview"></div>
  <div id="turns"></div>
  <div id="cands"></div>
  <label id="inlabel">Your turn</label>
  <textarea id="input" rows="3" placeholder="Enter = send · Shift+Enter = newline"></textarea>
  <div class="row" style="margin-top:8px">
    <button class="primary" id="send" onclick="sendTurn()">Send → model candidates</button>
    <button onclick="genFor('them')">Model turn</button>
    <button onclick="genFor('you')">Model writes my turn</button>
    <button onclick="undo()">Undo</button>
  </div>
</div></div>
<script>
const $=id=>document.getElementById(id); let turns=[]; let pending=null;
async function loadBackends(){const r=await fetch('/api/backends');const bs=await r.json();const s=$('backend');s.innerHTML='';
 bs.forEach(b=>{const o=document.createElement('option');o.value=b.key;o.textContent=`${b.key} — ${b.name}`;s.appendChild(o)});
 s.onchange=()=>{const b=bs.find(x=>x.key===s.value);$('binfo').textContent=b?b.checkpoint:''};s.onchange();}
function names(){return {you:$('you').value.trim()||'User',them:$('them').value.trim()||'Model'}}
function render(){const nm=names();$('hview').textContent=$('header').value;const t=$('turns');t.innerHTML='';
 turns.forEach((x,i)=>{const d=document.createElement('div');d.className='turn'+(x.speaker===nm.you?' you':'');
  d.innerHTML=`<span class="who"></span><span class="txt"></span><span class="tools"><button>edit</button><button>✕</button></span>`;
  d.querySelector('.who').textContent=x.speaker+':';d.querySelector('.txt').textContent=x.text;
  const [eb,xb]=d.querySelectorAll('.tools button');eb.onclick=()=>{const v=prompt('Edit turn',x.text);if(v!==null){x.text=v;render()}};
  xb.onclick=()=>{turns.splice(i,1);render()};t.appendChild(d)});$('inlabel').textContent=`${nm.you}'s turn`;}
function busy(b,msg){['send'].forEach(id=>$(id).disabled=b);$('status').textContent=msg||'ready'}
async function genFor(who){const nm=names();const speaker=who==='you'?nm.you:nm.them;busy(true,`generating ${speaker}…`);
 const steer=[];const vr=parseFloat($('vrep').value),v7=parseFloat($('v7').value);
 if(vr)steer.push({name:'vrep',alpha:vr});if(v7)steer.push({name:'v7-entropy',alpha:v7});
 try{const r=await fetch('/api/candidates',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
  backend:$('backend').value,header:$('header').value,turns,speaker,cast:[nm.you,nm.them],n:+$('n').value,
  max_tokens:+$('maxtok').value,temperature:+$('temp').value,repetition_penalty:+$('rep').value,steer})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);showCands(speaker,d.candidates);busy(false)}
 catch(e){busy(false,'error: '+e.message)}}
function showCands(speaker,cs){pending=speaker;const box=$('cands');box.innerHTML=`<div class="small">${speaker} — click to commit · <a href="#" id="reroll">reroll</a></div>`;
 box.querySelector('#reroll').onclick=e=>{e.preventDefault();genFor(speaker===names().you?'you':'them')};
 cs.forEach(c=>{const d=document.createElement('div');d.className='cand';
  const tags=(c.vetoes.length?c.vetoes.map(v=>`<span class="tag bad">${v}</span>`).join(''):'<span class="tag good">passes</span>')
   +`<span class="tag">${c.ended_by}</span>`+(c.cut_label?`<span class="tag">cut at ${c.cut_label}:</span>`:'');
  d.innerHTML=`<div>${tags}</div><div class="ctext"></div>`;d.querySelector('.ctext').textContent=c.text||'(empty)';
  d.onclick=()=>{if(!c.text)return;turns.push({speaker,text:c.text});box.innerHTML='';pending=null;render();$('input').focus()};box.appendChild(d)});}
function sendTurn(){const v=$('input').value.trim();const nm=names();if(v){turns.push({speaker:nm.you,text:v});$('input').value='';render()}genFor('them')}
function undo(){if($('cands').innerHTML){$('cands').innerHTML='';return}turns.pop();render()}
function resetAll(){if(confirm('Clear the transcript?')){turns=[];$('cands').innerHTML='';render()}}
function exportJson(){const row={header:$('header').value,turns};const blob=new Blob([JSON.stringify(row)+'\n'],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='transcript.jsonl';a.click()}
$('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendTurn()}});
['header','you','them'].forEach(id=>$(id).addEventListener('input',render));loadBackends();render();
</script></body></html>"""


def make_handler(backends: dict[str, str], client: Any) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info(fmt, *args)

        def _json(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                data = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/backends":
                self._json(200, [{"key": k, **backend_info(client, u)} for k, u in backends.items()])
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/api/candidates":
                self.send_error(404)
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                url = backends.get(req.get("backend", ""))
                if url is None:
                    self._json(400, {"error": f"unknown backend {req.get('backend')!r}"})
                    return
                self._json(200, {"candidates": candidates(client, url, req)})
            except Exception as e:  # report to the page instead of dropping the connection
                logger.exception("candidates failed")
                self._json(502, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def parse_backends(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in specs:
        key, sep, url = s.partition("=")
        if not sep or not key or not url.startswith("http"):
            raise SystemExit(f"--backend wants name=http://host:port, got {s!r}")
        out[key] = url
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import httpx

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [transcript-ui] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="web UI for plain-text transcripts against a base-mode server")
    ap.add_argument("--backend", action="append", required=True, help="name=http://host:port (repeatable)")
    ap.add_argument("--port", type=int, default=3344)
    ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args(argv)
    backends = parse_backends(a.backend)
    with httpx.Client() as client:
        server = http.server.ThreadingHTTPServer((a.bind, a.port), make_handler(backends, client))
        for k, u in backends.items():
            logger.info("backend %s -> %s (%s)", k, u, backend_info(client, u)["name"])
        logger.info("open http://%s:%d", a.bind, a.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
