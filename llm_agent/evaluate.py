"""Field-by-field scoring of predictions against hand-labelled gold answers."""
from schema import _norm

EXACT = ["event_type", "fuel", "zone", "status", "direction"]


def cap_ok(g, p):
    if g is None or p is None:
        return g is None and p is None
    return abs(g - p) <= max(5.0, 0.01 * g)


def date_ok(g, p):
    return (g[:10] if g else None) == (p[:10] if p else None)


def score(gold: list, pred: list) -> dict:
    """pred[i] is a record or None (failed). A failed record counts as wrong on every field."""
    n = len(gold)
    acc = {f: 0 for f in EXACT + ["capacity_mw", "start_time", "end_time", "key_fields_all_correct"]}
    for g, p in zip(gold, pred):
        if p is None:
            continue
        ok = {f: g[f] == p[f] for f in EXACT}
        ok["capacity_mw"] = cap_ok(g["capacity_mw"], p["capacity_mw"])
        ok["start_time"] = date_ok(g["start_time"], p["start_time"])
        ok["end_time"] = date_ok(g["end_time"], p["end_time"])
        ok["key_fields_all_correct"] = ok["event_type"] and ok["fuel"] and ok["direction"] and ok["capacity_mw"]
        for f, v in ok.items():
            acc[f] += int(v)
    out = {f: round(v / n, 3) for f, v in acc.items()}
    out["n"] = n
    out["answered"] = sum(p is not None for p in pred)
    return out


def grounded_rate(texts, pred):
    """Share of answered records whose evidence quote really appears in the source text."""
    ok = [(_norm(p["evidence"]) in _norm(t)) for t, p in zip(texts, pred) if p is not None]
    return round(sum(ok) / len(ok), 3) if ok else None
