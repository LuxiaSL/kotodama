"""posttrain.compliance: deterministic flags, PII redaction, turn hashes, scan -> blocklist -> assembly drop."""

from __future__ import annotations

import json

from kotodama.posttrain import compliance, loom


def test_classify_flags() -> None:
    # a co-occurrence rule: it over-flags innocent text like this, which is the intended trade at these rates
    assert "minor_explicit" in compliance.classify("The kids ran naked into the sea.")
    assert "minor_explicit" not in compliance.classify("The kids ran into the sea.")
    assert "minor_explicit" not in compliance.classify("He was naked in the sea.")
    assert compliance.classify("Write to mara.k@harbour-mail.org tonight.") == {"pii_email"}
    assert compliance.classify("Call me at (555) 867-5309.") == {"pii_phone"}
    assert compliance.classify("Her number was 078-05-1120.") == {"pii_ssn"}
    assert compliance.classify("Card 4111 1111 1111 1111 expires soon.") == {"pii_card"}
    assert "pii_card" not in compliance.classify("Order 4111 1111 1111 1112 shipped.")  # fails the Luhn check
    assert compliance.classify("The server at 192.168.0.12 is down.") == {"pii_ip"}
    assert compliance.classify("It was 1987, page 300, aisle 12.") == set()


def test_redact_pii() -> None:
    out = compliance.redact_pii("Mail mara@harbour.org or call 555-867-5309; card 4111-1111-1111-1111.")
    assert out == "Mail <email> or call <phone>; card <card>."


def test_hash_blocklist_scan_and_assembly(tmp_path) -> None:
    assert compliance.turn_hash("Hello  There") == compliance.turn_hash("hello there")
    run = tmp_path / "run"
    run.mkdir()
    bad = "The kids ran naked into the sea."
    rows = [{"tree": "t0", "leaf": "t0/r.0.0.0", "depth": 3, "n_seed": 0, "header": "H", "cast": ["A", "B"],
             "turns": [{"speaker": "A", "text": "Morning, all."}, {"speaker": "B", "text": bad},
                       {"speaker": "A", "text": "Mail me at a@b.co."}]},
            {"tree": "t1", "leaf": "t1/r.0.0", "depth": 2, "n_seed": 0, "header": "H", "cast": ["A", "B"],
             "turns": [{"speaker": "A", "text": "Evening."}, {"speaker": "B", "text": "Write to a@b.co."}]}]
    (run / "paths.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "trees.jsonl").write_text("".join(json.dumps({"tree": t, "seed": t}) + "\n" for t in ("t0", "t1")))
    out = tmp_path / "block.tsv"
    assert compliance.main(["scan", str(run), "--out", str(out)]) == 0
    text = out.read_text()
    assert bad not in text and compliance.turn_hash(bad) in text  # hashes only, never the text
    assert compliance.load_blocklist(out) == {compliance.turn_hash(bad)}
    docs_path = tmp_path / "docs.jsonl"
    assert loom.main(["assemble", str(run), "--out", str(docs_path), "--min_new", "1", "--blocklist", str(out),
                      "--redact_pii"]) == 0
    docs = [json.loads(x) for x in docs_path.read_text().splitlines()]
    texts = [t["text"] for d in docs for t in d["turns"]]
    assert bad not in texts and "Write to <email>." in texts  # t0 dropped whole; t1 kept, redacted
