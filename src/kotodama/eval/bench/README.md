# The binding benchmark

The benchmark scores whether a language model keeps track of who holds what, who *I* and *you* refer to, and whether
it can learn a new rule from the examples in its prompt. Every battery is a frozen, sha256-pinned set of prompts, and
every model is scored the same way: by the log-probability it gives each candidate answer.

## Where the pieces are

| piece | module |
|---|---|
| registry: construct, recipe, clean readings, audit flags per eval | `kotodama.eval.bench.registry` |
| the one import path into the rest of kotodama (tokenizer, ChatML, `koto:` checkpoints) | `kotodama.eval.bench.host` |
| pinned batteries, prompt-file checks, the suite CLI and readers | `kotodama.eval.binding` (`batteries.py`, `__main__.py`, `readers.py`) |
| bindprobes families and their item generator | `kotodama.eval.bindprobes` |
| riders: the box-tracking grid and its in-loop slice | `kotodama.eval.riders` (`lite.py` for the slice) |
| the closed-set scorer | `kotodama.eval.binding.score` (prompt rows), `kotodama.eval.riders.score` (shared with the server's `/logprobs` and the in-loop evals) |

The four packages import nothing from kotodama outside themselves except through `host.py`;
`tests/test_bench_boundary.py` enforces this, and checks that every pinned battery has a registry row.

## Reading a score

Each registry row names its **clean readings**: the conditions that count when two models are compared. For the
bindprobes families these are `0shot` (bare items, no worked examples; the floor) and `fair` (four worked examples
balanced on every cell component). Other conditions stay available as diagnostics. A **flag** marks a reading that
is not a clean measure of its construct on the panel it names, for example a few-shot condition whose exemplars
can be copied. A flag is a finding about the reading, not about the model.

An eval changes only on validity grounds: a rendering that moves the score, an off-construct lever that moves it,
or an item-level shortcut. A new rendering has to reduce the spread across renderings, not raise the score. A new
version of a battery is a new name and a new pin beside the old one.

## Commands

```
python -m kotodama.eval.binding list                                       # batteries and named groups
python -m kotodama.eval.binding run --model hf:Qwen/Qwen3-1.7B-Base --name qwen3-1.7b --batteries scorecard --out scores/
python -m kotodama.eval.binding report --scores scores/                   # the scorecard of the prompt-row batteries
python -m kotodama.eval.binding gen-capacity --out capacity-v1.jsonl      # regenerates the pinned file byte for byte
python -m kotodama.eval.binding gen-fluid --out DIR
python -m kotodama.eval.binding gen-formupdate --out DIR
```

Prompt files live under one directory (`--prompts-dir`, or `$KOTODAMA_BINDING_PROMPTS`); a run refuses a file
whose bytes, or whose rendered prompts, differ from the pin.
