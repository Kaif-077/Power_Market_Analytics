"""Rule-based baseline: keywords and regular expressions only. The LLM agent has to beat this to be worth using."""
import re
from datetime import date

COUNTRIES = {"germany": "DE", "german": "DE", "france": "FR", "french": "FR", "spain": "ES", "spanish": "ES",
             "italy": "IT", "italian": "IT", "netherlands": "NL", "dutch": "NL", "belgium": "BE", "belgian": "BE",
             "poland": "PL", "polish": "PL", "czech": "CZ", "norway": "NO", "norwegian": "NO", "sweden": "SE",
             "swedish": "SE", "denmark": "DK", "danish": "DK", "britain": "GB", "uk": "GB", "finland": "FI",
             "finnish": "FI", "austria": "AT", "switzerland": "CH"}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
_MON = "|".join(MONTHS)
DATE_RE = re.compile(rf"(\d{{1,2}})(?:\s+to\s+(\d{{1,2}}))?\s+({_MON})\s+(\d{{4}})", re.I)
CAP_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(GW|MW)\b")

FUEL_RULES = [("lignite", ["lignite"]), ("coal", ["coal"]), ("nuclear", ["nuclear", "reactor"]), ("other", ["biomass"]),
              ("hydro", ["hydro", "reservoir", "water level"]), ("wind", ["wind", "turbine"]), ("solar", ["solar"]),
              ("grid", ["kv", " line", "link", "cable", "interconnector", "hvdc", "grid operator"]),
              ("gas", ["gas", "lng", "pipeline", "ttf"])]
OUTAGE = ["outage", "shut", "trip", "maintenance", "unavailable", "refuelling", "cut output", "repairs", "out of service",
          "returned to service", "fault"]


def _has(t, words):
    return any(w in t for w in words)


def _dates(text):
    out = []
    for m in DATE_RE.finditer(text):
        d1, d2, mon, yr = int(m[1]), m[2], MONTHS[m[3].lower()], int(m[4])
        out.append((date(yr, mon, d1).isoformat(), m.start()))
        if d2:
            out.append((date(yr, mon, int(d2)).isoformat(), m.start() + 1))
    return out


def extract_baseline(text: str) -> dict:
    t = text.lower()
    fuel = next((f for f, ws in FUEL_RULES if _has(t, ws)), "none")
    zone = None
    hits = [(t.find(k), v) for k, v in COUNTRIES.items() if re.search(rf"\b{k}\b", t)]
    if hits:
        zone = min(hits)[1]
    cap = None
    m = CAP_RE.search(text)
    if m:
        cap = float(m[1].replace(",", "")) * (1000 if m[2] == "GW" else 1)

    status = ("unplanned" if _has(t, ["unplanned", "unexpected", "tripped", "automatically", "fault", "forced", "unscheduled"])
              else "resolved" if _has(t, ["returned to service", "back online", "restarted"])
              else "planned" if _has(t, ["planned", "scheduled", "maintenance", "refuelling"]) else "unknown")
    if fuel == "grid" and _has(t, OUTAGE):
        etype = "grid_outage"
    elif fuel in ("gas", "coal", "lignite", "nuclear", "wind", "solar", "hydro", "other") and _has(t, OUTAGE):
        etype = "plant_outage"
    elif fuel == "gas" and not _has(t, ["plant", "fired", "combined-cycle"]):
        etype = "gas_supply"
    elif fuel in ("wind", "solar", "hydro") or _has(t, ["storm", "heatwave", "forecast"]):
        etype = "weather"
    elif _has(t, ["demand", "consumption"]):
        etype = "demand"
    elif _has(t, ["government", "parliament", "subsidy", "regulation", "law ", "cap ", "auction", "tax"]):
        etype = "policy"
    else:
        etype = "other"

    if status == "resolved":
        direction = "down"
    elif etype in ("plant_outage", "grid_outage"):
        direction = "up"
    elif etype == "gas_supply":
        direction = "down" if _has(t, ["easing", "eases"]) else "up"
    elif etype == "weather":
        direction = "down" if _has(t, ["strong wind", "record", "negative"]) else "up" if _has(t, ["fell short", "dry", "below normal"]) else "neutral"
    elif etype == "demand":
        direction = "down" if _has(t, ["drop", "fall", "decline", "lower"]) else "up" if _has(t, ["rise", "higher", "boost", "surge"]) else "neutral"
    else:
        direction = "neutral"

    ds = [d for d, _ in _dates(text)]
    start = end = None
    if len(ds) >= 2:
        start, end = ds[0], ds[1]
    elif len(ds) == 1:
        before = t[max(0, _dates(text)[0][1] - 25):_dates(text)[0][1]]
        if status == "resolved" or _has(before, ["until", "by ", "return"]):
            end = ds[0]
        else:
            start = ds[0]
    return {"event_type": etype, "fuel": fuel, "zone": zone, "capacity_mw": cap, "status": status, "direction": direction,
            "start_time": start, "end_time": end, "confidence": 0.5, "evidence": " ".join(text.split()[:8])}
