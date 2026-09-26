"""kotalk — interactive chat / completion CLI for kotodama models (stdlib only).

The gateway (one port, default :2222) hosts both models and does the routing,
so kotalk just picks a surface and the model follows:

  complete   raw continuation REPL for the BASE model: a growing text buffer
             you extend, sample N candidates from, and branch.
  chat       turn-based conversation with the INSTRUCT model (server-side
             ChatML). Default surface.

Transport is the server's native /generate (proxied by the gateway), which keeps
full sampling control — top_k, repetition_penalty, stop strings, token
streaming — that the OpenAI endpoints don't expose. The target model is selected
with the request's "model" field; the gateway routes on it. (A bare server
ignores the field, so this also works with a single server.)

Usage:
  python -m kotodama.serve.chat                                # chat surface, $KOTODAMA_ENDPOINT
  python -m kotodama.serve.chat complete                       # base continuation REPL
  python -m kotodama.serve.chat complete --once "The capital of France is"
  python -m kotodama.serve.chat chat --load sessions/foo.json
  python -m kotodama.serve.chat chat --model <id> --law chat   # pin a model id / sampling law

Design notes:
  - stdlib only; streams token deltas for n=1, candidate-pick for n>1.
  - defaults = the "chat" law in kotodama.serve.laws (temp 0.9 / top_k 0 /
    rep-pen 1.2 / top_p 0 — pure temperature); --law picks another.
  - /raw shows the exact prompt string sent (complete surface).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from kotodama.serve.laws import LAWS, law

try:
    import readline  # noqa: F401  — line editing + input history
except ImportError:
    pass

DEFAULT_ENDPOINT = os.environ.get("KOTODAMA_ENDPOINT", "http://localhost:2222")
SESSIONS_DIR = Path(os.environ.get("KOTODAMA_SESSIONS_DIR", Path.cwd() / "sessions"))


# ── colors ───────────────────────────────────────────────────────────────────

class C:
    on = sys.stdout.isatty()
    @classmethod
    def _c(cls, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if cls.on else s
    @classmethod
    def dim(cls, s: str) -> str: return cls._c("90", s)
    @classmethod
    def blue(cls, s: str) -> str: return cls._c("94", s)
    @classmethod
    def green(cls, s: str) -> str: return cls._c("92", s)
    @classmethod
    def yellow(cls, s: str) -> str: return cls._c("93", s)
    @classmethod
    def red(cls, s: str) -> str: return cls._c("91", s)
    @classmethod
    def cyan(cls, s: str) -> str: return cls._c("96", s)
    @classmethod
    def bold(cls, s: str) -> str: return cls._c("1", s)


# ── sampling state ───────────────────────────────────────────────────────────

@dataclass
class Sampling:
    temperature: float = 0.9
    max_tokens: int = 256
    top_p: float = 0.0          # 0.0 = disabled — pure temperature (kotodama law)
    top_k: int = 0              # 0 = no truncation (k50 ≡ none; SERVING-LAWS 2026-07-05)
    rep_penalty: float = 1.2    # chat law (i-don't-know-collapse fix; gamut-confirmed 2026-07-09)
    n: int = 1
    stop: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"temp={self.temperature}", f"max={self.max_tokens}",
                f"top_k={self.top_k}", f"rep={self.rep_penalty}", f"n={self.n}"]
        bits.append(f"top_p={self.top_p}" if self.top_p > 0 else "top_p=off")
        if self.stop:
            bits.append(f"stop={self.stop}")
        return " ".join(bits)


# ── client ───────────────────────────────────────────────────────────────────

class ServerError(RuntimeError):
    pass


@dataclass
class GenResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tps: float = 0.0


class KotodamaClient:
    """Thin client over the gateway's /generate (+ /v1/models). The `model`
    field routes to a pool; the gateway dispatches/load-balances from there."""

    def __init__(self, endpoint: str, model: Optional[str] = None,
                 timeout: float = 300.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.extra: dict = {}  # merged into every /generate payload (e.g. steering)
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, payload: Optional[dict] = None) -> Any:
        req = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read())
                detail = body.get("detail") or body.get("error") or ""
            except Exception:
                detail = ""
            raise ServerError(f"HTTP {e.code} {path}: {detail or e.reason}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ServerError(f"{path}: {e}") from e

    def list_models(self) -> list[str]:
        d = self._request("/v1/models")
        return [m["id"] for m in d.get("data", []) if "id" in m]

    def info(self) -> dict:
        """serve.py /info — model params/mode/checkpoint. Used by kotoagent `new`."""
        return self._request("/info")

    def _payload(self, s: Sampling, prompt: Optional[str],
                 messages: Optional[list[dict[str, str]]], stream: bool) -> dict:
        p: dict[str, Any] = {
            "max_new_tokens": s.max_tokens,
            "temperature": s.temperature,
            "top_k": s.top_k,
            "top_p": s.top_p,
            "repetition_penalty": s.rep_penalty,
            "stop_strings": s.stop,
            "stream": stream,
        }
        if self.model:
            p["model"] = self.model
        if messages is not None:
            p["messages"] = messages
        else:
            p["prompt"] = prompt
        p.update(self.extra)
        return p

    def generate(self, s: Sampling, prompt: Optional[str] = None,
                 messages: Optional[list[dict[str, str]]] = None) -> GenResult:
        d = self._request("/generate", self._payload(s, prompt, messages, stream=False))
        return GenResult(text=d.get("text", ""),
                         prompt_tokens=d.get("prompt_tokens", 0),
                         completion_tokens=d.get("completion_tokens", 0),
                         tps=d.get("tokens_per_second", 0.0))

    def generate_stream(self, s: Sampling, prompt: Optional[str] = None,
                        messages: Optional[list[dict[str, str]]] = None,
                        ) -> Iterator[tuple[str, Any]]:
        """Yields ("token", delta) then ("done", info_dict). SSE over /generate."""
        req = urllib.request.Request(
            f"{self.endpoint}/generate",
            data=json.dumps(self._payload(s, prompt, messages, stream=True)).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise ServerError(data["error"])
                    if data.get("done"):
                        yield ("done", data)
                        return
                    if "token" in data:
                        yield ("token", data["token"])
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ServerError(f"/generate stream: {e}") from e


# ── sessions ─────────────────────────────────────────────────────────────────

def save_session(kind: str, endpoint: str, model: str, sampling: Sampling,
                 state: dict[str, Any], filename: Optional[str]) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SESSIONS_DIR / f"kotalk_{kind}_{ts}.json"
    else:
        path = Path(filename)
        if not path.is_absolute() and not path.parent.name:
            path = SESSIONS_DIR / path
    payload = {"kind": kind, "endpoint": endpoint, "model": model,
               "saved": datetime.now().isoformat(timespec="seconds"),
               "sampling": asdict(sampling), **state}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def load_session(filename: str) -> dict[str, Any]:
    path = Path(filename)
    if not path.exists() and not path.is_absolute():
        alt = SESSIONS_DIR / filename
        path = alt if alt.exists() else path
    return json.loads(path.read_text())


# ── REPL base ────────────────────────────────────────────────────────────────

class Repl:
    KIND = "?"

    def __init__(self, client: KotodamaClient, sampling: Sampling,
                 model_name: str) -> None:
        self.client = client
        self.s = sampling
        self.model_name = model_name
        self.show_raw = False
        self.last_stats: Optional[GenResult] = None
        # command -> (handler, help). Subclasses extend.
        self.commands: dict[str, tuple[Any, str]] = {
            "/help": (self.cmd_help, "show commands"),
            "/exit": (self.cmd_exit, "save session and quit"),
            "/quit": (self.cmd_exit, "alias of /exit"),
            "/save": (self.cmd_save, "/save [file] — save session JSON"),
            "/temp": (self.cmd_temp, "/temp <v> — temperature"),
            "/maxtok": (self.cmd_maxtok, "/maxtok <v> — max new tokens"),
            "/topp": (self.cmd_topp, "/topp <v> — top_p (0 = off, the default)"),
            "/topk": (self.cmd_topk, "/topk <v> — top_k"),
            "/rep": (self.cmd_rep, "/rep <v> — repetition penalty (1.0 = off)"),
            "/n": (self.cmd_n, "/n <k> — candidates per turn (pick one)"),
            "/stop": (self.cmd_stop, "/stop [s1|s2|…] — extra stop strings (empty clears)"),
            "/settings": (self.cmd_settings, "show sampling settings"),
            "/stats": (self.cmd_stats, "tok/s + token counts of last generation"),
            "/raw": (self.cmd_raw, "toggle showing the exact prompt sent"),
            "/model": (self.cmd_model, "show this session's model + available models"),
            "/steer": (self.cmd_steer, "/steer [name[:a] [+ name[:a]...]] | off — "
                       "server-side steering vectors (steered servers only)"),
        }

    # — generic command handlers —
    def cmd_help(self, arg: str) -> None:
        print()
        for cmd, (_h, doc) in self.commands.items():
            print(f"  {C.green(f'{cmd:<10}')} {C.dim(doc)}")
        print()

    def cmd_exit(self, arg: str) -> None:
        path = self._save(None)
        print(C.yellow(f"saved → {path}"))
        sys.exit(0)

    def cmd_save(self, arg: str) -> None:
        print(C.green(f"saved → {self._save(arg or None)}"))

    def _set_float(self, attr: str, arg: str, cast=float) -> None:
        if not arg:
            print(C.yellow(f"{attr} = {getattr(self.s, attr)}"))
            return
        try:
            setattr(self.s, attr, cast(arg))
            print(C.green(f"{attr} = {getattr(self.s, attr)}"))
        except ValueError:
            print(C.red(f"bad value: {arg}"))

    def cmd_temp(self, a: str) -> None: self._set_float("temperature", a)
    def cmd_maxtok(self, a: str) -> None: self._set_float("max_tokens", a, int)
    def cmd_topp(self, a: str) -> None: self._set_float("top_p", a)
    def cmd_topk(self, a: str) -> None: self._set_float("top_k", a, int)
    def cmd_rep(self, a: str) -> None: self._set_float("rep_penalty", a)
    def cmd_n(self, a: str) -> None: self._set_float("n", a, int)

    def cmd_stop(self, arg: str) -> None:
        self.s.stop = [s for s in arg.split("|") if s] if arg else []
        print(C.green(f"stop = {self.s.stop or '(none)'}"))

    def cmd_settings(self, arg: str) -> None:
        print(C.yellow(self.s.summary()))

    def cmd_stats(self, arg: str) -> None:
        r = self.last_stats
        if r is None:
            print(C.yellow("no generation yet"))
        else:
            print(C.yellow(f"prompt={r.prompt_tokens} tok · completion={r.completion_tokens} tok · {r.tps:.1f} tok/s"))

    def cmd_raw(self, arg: str) -> None:
        self.show_raw = not self.show_raw
        print(C.green(f"raw prompt display {'ON' if self.show_raw else 'OFF'}"))

    def cmd_model(self, arg: str) -> None:
        line = f"this session → {C.bold(self.model_name)}"
        try:
            avail = self.client.list_models()
            line += f"   {C.dim('available: ' + ', '.join(avail))}"
        except ServerError as e:
            line += f"   {C.red(str(e))}"
        print(line)

    def cmd_steer(self, arg: str) -> None:
        arg = arg.strip()
        if not arg:
            stack = self.client.extra.get("vectors")
            print(f"steering: {stack if stack else C.dim('off')}")
            try:
                info = self.client.info()
                al = (info.get("aliases")
                      or (info.get("steering") or {}).get("aliases") or {})
                if al:
                    print(C.dim("available: " + ", ".join(
                        f"{k}(α{v.get('alpha')})" for k, v in sorted(al.items()))))
            except ServerError as e:
                print(C.red(f"(no steering info: {e})"))
            return
        if arg in ("off", "none", "clear"):
            self.client.extra.pop("vectors", None)
            print("steering off")
            return
        stack = []
        try:
            for part in arg.split("+"):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    name, a = part.rsplit(":", 1)
                    stack.append({"name": name.strip(), "alpha": float(a)})
                else:
                    stack.append({"name": part})
        except ValueError as e:
            print(C.red(f"bad /steer expression: {e}"))
            return
        self.client.extra["vectors"] = stack
        print("steering: " + " + ".join(
            f"{s['name']}" + (f":{s['alpha']}" if "alpha" in s else "") for s in stack))

    # — shared generation plumbing —
    def _gen_stream_print(self, prompt: Optional[str] = None,
                          messages: Optional[list[dict[str, str]]] = None,
                          prefix: str = "") -> str:
        """Stream one generation, printing deltas. Returns full text."""
        if prefix:
            print(prefix, end="", flush=True)
        full = ""
        done: dict[str, Any] = {}
        for kind, val in self.client.generate_stream(self.s, prompt=prompt,
                                                     messages=messages):
            if kind == "token":
                print(val, end="", flush=True)
                full += val
            else:
                done = val
        print()
        self.last_stats = GenResult(full, done.get("prompt_tokens", 0),
                                    done.get("completion_tokens", 0))
        return full

    def _gen_candidates(self, prompt: Optional[str] = None,
                        messages: Optional[list[dict[str, str]]] = None,
                        ) -> list[str]:
        out: list[str] = []
        for i in range(self.s.n):
            print(C.dim(f"  [{i + 1}/{self.s.n}]…"), end="\r", flush=True)
            r = self.client.generate(self.s, prompt=prompt, messages=messages)
            self.last_stats = r
            out.append(r.text)
        print(" " * 20, end="\r")
        return out

    def _pick(self, candidates: list[str]) -> Optional[str]:
        for i, c in enumerate(candidates):
            print(f"\n{C.cyan(f'[{i + 1}]')} {c}")
        print()
        while True:
            try:
                choice = input(C.yellow(f"pick 1-{len(candidates)}, r=reroll, c=cancel> ")).strip().lower()
            except (KeyboardInterrupt, EOFError):
                return None
            if choice == "c":
                return None
            if choice == "r":
                return "__REROLL__"
            try:
                k = int(choice) - 1
                if 0 <= k < len(candidates):
                    return candidates[k]
            except ValueError:
                pass
            print(C.red("?"))

    # — overridables —
    def _save(self, filename: Optional[str]) -> Path:
        raise NotImplementedError

    def banner(self) -> None:
        print(f"\n{C.cyan('─' * 64)}")
        print(C.bold(f"kotalk · {self.KIND} · {self.model_name}"))
        print(C.dim(f"endpoint {self.client.endpoint}"))
        print(C.dim(self.s.summary()))
        print(C.dim("/help for commands"))
        print(f"{C.cyan('─' * 64)}\n")

    def dispatch(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        entry = self.commands.get(parts[0].lower())
        if entry is None:
            print(C.red(f"unknown command {parts[0]} — /help"))
            return True
        entry[0](parts[1] if len(parts) > 1 else "")
        return True

    def loop(self) -> None:
        self.banner()
        while True:
            try:
                line = input(self.prompt_str())
            except KeyboardInterrupt:
                print(C.yellow("\n(/exit to save & quit)"))
                continue
            except EOFError:
                self.cmd_exit("")
            line = line.rstrip("\n")
            if line.startswith("/"):
                self.dispatch(line.strip())
                continue
            try:
                self.step(line)
            except ServerError as e:
                print(C.red(f"server error: {e}"))
            except KeyboardInterrupt:
                print(C.yellow("\n(generation interrupted)"))

    def prompt_str(self) -> str:
        raise NotImplementedError

    def step(self, line: str) -> None:
        raise NotImplementedError


# ── complete surface ─────────────────────────────────────────────────────────

class CompleteRepl(Repl):
    """Growing-buffer continuation REPL for the base model."""
    KIND = "complete"

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.buffer = ""
        self.undo: list[str] = []
        self.commands.update({
            "/show": (self.cmd_show, "print the full buffer"),
            "/clear": (self.cmd_clear, "reset the buffer"),
            "/pop": (self.cmd_pop, "undo the last append (input or generation)"),
            "/ml": (self.cmd_ml, "multiline input — finish with a lone '.'"),
            "/load": (self.cmd_load, "/load <file> — restore a saved session"),
        })

    def prompt_str(self) -> str:
        tag = f"{len(self.buffer)}ch" if self.buffer else "empty"
        return C.blue(f"text[{tag}]> ")

    def _push(self, text: str) -> None:
        self.undo.append(self.buffer)
        del self.undo[:-50]
        self.buffer += text

    def cmd_show(self, arg: str) -> None:
        print(f"\n{self.buffer or C.dim('(empty)')}\n")

    def cmd_clear(self, arg: str) -> None:
        self._push("")
        self.buffer = ""
        print(C.green("buffer cleared"))

    def cmd_pop(self, arg: str) -> None:
        if self.undo:
            self.buffer = self.undo.pop()
            print(C.green(f"popped — buffer now {len(self.buffer)} chars"))
        else:
            print(C.yellow("nothing to pop"))

    def cmd_ml(self, arg: str) -> None:
        print(C.dim("multiline — end with a lone '.'"))
        lines: list[str] = []
        while True:
            try:
                ln = input()
            except (KeyboardInterrupt, EOFError):
                print(C.yellow("(cancelled)"))
                return
            if ln.strip() == ".":
                break
            lines.append(ln)
        self.step("\n".join(lines))

    def cmd_load(self, arg: str) -> None:
        if not arg:
            print(C.red("usage: /load <file>"))
            return
        try:
            d = load_session(arg)
            self.buffer = d.get("buffer", "")
            print(C.green(f"loaded buffer ({len(self.buffer)} chars)"))
        except (OSError, json.JSONDecodeError) as e:
            print(C.red(f"load failed: {e}"))

    def _save(self, filename: Optional[str]) -> Path:
        return save_session(self.KIND, self.client.endpoint, self.model_name,
                            self.s, {"buffer": self.buffer}, filename)

    def step(self, line: str) -> None:
        if line:
            self._push(line)
        if not self.buffer:
            print(C.yellow("buffer is empty — type a prefix first"))
            return
        if self.show_raw:
            print(C.dim(f"--- prompt ---\n{self.buffer}\n--------------"))
        if self.s.n <= 1:
            text = self._gen_stream_print(prompt=self.buffer, prefix=C.green("· "))
            self._push(text)
        else:
            while True:
                picked = self._pick(self._gen_candidates(prompt=self.buffer))
                if picked == "__REROLL__":
                    continue
                if picked is not None:
                    self._push(picked)
                break


# ── chat surface ─────────────────────────────────────────────────────────────

class ChatRepl(Repl):
    """Turn-based conversation with the instruct model (server-side ChatML)."""
    KIND = "chat"

    def __init__(self, client: KotodamaClient, sampling: Sampling,
                 model_name: str) -> None:
        super().__init__(client, sampling, model_name)
        self.history: list[dict[str, str]] = []
        self.system: Optional[str] = None
        self.commands.update({
            "/history": (self.cmd_history, "show the conversation"),
            "/clear": (self.cmd_clear, "clear the conversation"),
            "/undo": (self.cmd_undo, "remove the last exchange"),
            "/retry": (self.cmd_retry, "regenerate the last reply"),
            "/system": (self.cmd_system, "/system <text> — set system prompt (empty clears)"),
            "/ml": (self.cmd_ml, "multiline input — finish with a lone '.'"),
            "/load": (self.cmd_load, "/load <file> — restore a saved session"),
        })

    def prompt_str(self) -> str:
        return C.blue("you> ")

    # — commands —
    def cmd_history(self, arg: str) -> None:
        if not self.history:
            print(C.yellow("(no turns yet)"))
        if self.system:
            print(f"\n{C.cyan('system:')} {self.system}")
        for m in self.history:
            tag = C.blue("you:") if m["role"] == "user" else C.green("model:")
            print(f"\n{tag} {m['content']}")
        print()

    def cmd_clear(self, arg: str) -> None:
        self.history = []
        print(C.green("conversation cleared"))

    def cmd_undo(self, arg: str) -> None:
        while self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        print(C.green(f"undone — {len(self.history)} messages left"))

    def cmd_retry(self, arg: str) -> None:
        if self.history and self.history[-1]["role"] == "assistant":
            self.history.pop()
            try:
                self._respond()
            except ServerError as e:
                print(C.red(f"server error: {e}"))
        else:
            print(C.yellow("nothing to retry"))

    def cmd_system(self, arg: str) -> None:
        self.system = arg or None
        print(C.green(f"system prompt {'set' if self.system else 'cleared'}"))

    def cmd_ml(self, arg: str) -> None:
        print(C.dim("multiline — end with a lone '.'"))
        lines: list[str] = []
        while True:
            try:
                ln = input()
            except (KeyboardInterrupt, EOFError):
                print(C.yellow("(cancelled)"))
                return
            if ln.strip() == ".":
                break
            lines.append(ln)
        self.step("\n".join(lines))

    def cmd_load(self, arg: str) -> None:
        if not arg:
            print(C.red("usage: /load <file>"))
            return
        try:
            d = load_session(arg)
            self.history = d.get("history", [])
            self.system = d.get("system")
            print(C.green(f"loaded {len(self.history)} messages"))
        except (OSError, json.JSONDecodeError) as e:
            print(C.red(f"load failed: {e}"))

    def _save(self, filename: Optional[str]) -> Path:
        return save_session(self.KIND, self.client.endpoint, self.model_name,
                            self.s, {"history": self.history, "system": self.system},
                            filename)

    # — generation —
    def _messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.extend(self.history)
        return msgs

    def _respond(self) -> None:
        messages = self._messages()
        if self.s.n <= 1:
            text = self._gen_stream_print(messages=messages, prefix=C.green("model> ")).strip()
        else:
            cands = [c.strip() for c in self._gen_candidates(messages=messages)]
            while True:
                picked = self._pick(cands)
                if picked == "__REROLL__":
                    cands = [c.strip() for c in self._gen_candidates(messages=messages)]
                    continue
                break
            if picked is None:
                if self.history and self.history[-1]["role"] == "user":
                    self.history.pop()
                return
            text = picked
        self.history.append({"role": "assistant", "content": text})

    def step(self, line: str) -> None:
        if not line:
            # empty input: let the model take the next turn unprompted
            self._respond()
            return
        self.history.append({"role": "user", "content": line})
        self._respond()


# ── entry ────────────────────────────────────────────────────────────────────

def resolve_models(model_ids: list[str]) -> tuple[str, str]:
    """(base_model_id, chat_model_id) from the gateway's /v1/models list."""
    if len(model_ids) == 1:
        return model_ids[0], model_ids[0]
    base = next((m for m in model_ids if "base" in m.lower()), model_ids[0])
    chat = next((m for m in model_ids if "instruct" in m.lower() or "chat" in m.lower()),
                model_ids[0])
    return base, chat


def main() -> None:
    ap = argparse.ArgumentParser(
        description="kotalk — kotodama base & instruct CLI (via the gateway)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("surface", nargs="?", default="chat",
                    choices=["chat", "complete"],
                    help="chat = instruct model, complete = base continuation buffer")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="gateway or single server base URL (env KOTODAMA_ENDPOINT)")
    ap.add_argument("--law", default="chat", choices=sorted(LAWS),
                    help="sampling law (kotodama.serve.laws); --temp/--top-k/... override it")
    ap.add_argument("--model", default=None,
                    help="pin a model id explicitly (default: picked by surface from /v1/models)")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-p", type=float, default=None, help="0 = pure temperature")
    ap.add_argument("--top-k", type=int, default=None, help="0 = no truncation")
    ap.add_argument("--rep", type=float, default=None, help="repetition penalty")
    ap.add_argument("--n", type=int, default=1, help="candidates per turn")
    ap.add_argument("--load", default=None, help="session JSON to restore")
    ap.add_argument("--once", default=None, metavar="TEXT",
                    help="non-interactive: one generation, print, exit")
    args = ap.parse_args()

    endpoint = args.endpoint
    base = law(args.law)
    pick = lambda given, key: base[key] if given is None else given  # noqa: E731
    sampling = Sampling(temperature=pick(args.temp, "temperature"), max_tokens=args.max_tokens,
                        top_p=pick(args.top_p, "top_p"), top_k=int(pick(args.top_k, "top_k")),
                        rep_penalty=pick(args.rep, "repetition_penalty"), n=args.n)

    disco = KotodamaClient(endpoint)
    try:
        model_ids = disco.list_models()
    except ServerError as e:
        print(C.red(f"cannot reach server at {endpoint}: {e}"))
        sys.exit(1)
    if not model_ids:
        print(C.red(f"server at {endpoint} returned no models"))
        sys.exit(1)

    base_model, chat_model = resolve_models(model_ids)
    if args.surface == "complete":
        model = args.model or base_model
    else:
        model = args.model or chat_model
    client = KotodamaClient(endpoint, model=model)

    if args.surface == "complete":
        repl: Repl = CompleteRepl(client, sampling, model)
    else:
        repl = ChatRepl(client, sampling, model)

    if args.load:
        repl.commands["/load"][0](args.load)

    if args.once is not None:
        if isinstance(repl, CompleteRepl):
            repl.buffer = args.once
            print(client.generate(sampling, prompt=repl.buffer).text)
        else:
            repl.history.append({"role": "user", "content": args.once})
            print(client.generate(sampling, messages=repl._messages()).text.strip())
        return

    repl.loop()


if __name__ == "__main__":
    main()
