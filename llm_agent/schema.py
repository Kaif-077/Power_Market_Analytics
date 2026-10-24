"""What the agent must extract from a news item or outage notice, and the rules a valid answer must satisfy."""
import re

EVENT_TYPES = ["plant_outage", "grid_outage", "gas_supply", "weather", "demand", "policy", "other"]
FUELS = ["gas", "coal", "lignite", "nuclear", "wind", "solar", "hydro", "grid", "none", "other"]
STATUSES = ["planned", "unplanned", "resolved", "unknown"]
DIRECTIONS = ["up", "down", "neutral"]  # expected effect on wholesale power prices, as the text implies
KEYS = ["event_type", "fuel", "zone", "capacity_mw", "status", "direction", "start_time", "end_time",
        "confidence", "evidence"]
MAX_PLAUSIBLE_MW = 20000
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def validate(rec, source_text: str) -> list:
    """Return a list of problems (empty list = valid). The messages are fed back to the model on a retry."""
    if not isinstance(rec, dict):
        return ["Answer must be a single JSON object."]
    errs = [f"Missing key: {k}" for k in KEYS if k not in rec]
    if errs:
        return errs
    for key, allowed in (("event_type", EVENT_TYPES), ("fuel", FUELS), ("status", STATUSES), ("direction", DIRECTIONS)):
        if rec[key] not in allowed:
            errs.append(f"{key} must be one of {allowed}, got {rec[key]!r}.")
    z = rec["zone"]
    if z is not None and not (isinstance(z, str) and re.fullmatch(r"[A-Z]{2}", z)):
        errs.append("zone must be a two-letter country code such as 'DE', or null if no country is given.")
    c = rec["capacity_mw"]
    if c is not None:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or c < 0 or c > MAX_PLAUSIBLE_MW:
            errs.append(f"capacity_mw must be null or a number between 0 and {MAX_PLAUSIBLE_MW} (in MW; convert GW).")
    times = {}
    for k in ("start_time", "end_time"):
        v = rec[k]
        if v is not None and not (isinstance(v, str) and _DATE.match(v)):
            errs.append(f"{k} must be null or 'YYYY-MM-DD' (optionally 'THH:MM').")
        elif v is not None:
            times[k] = v
    if len(times) == 2 and times["end_time"][:10] < times["start_time"][:10]:
        errs.append("end_time is before start_time.")
    conf = rec["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        errs.append("confidence must be a number between 0 and 1.")
    ev = rec["evidence"]
    if not isinstance(ev, str) or not ev.strip():
        errs.append("evidence must be a short exact quote from the text.")
    elif _norm(ev) not in _norm(source_text):
        errs.append("evidence must be copied exactly from the text; this quote is not in it.")
    return errs
