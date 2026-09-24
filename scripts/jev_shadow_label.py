#!/usr/bin/env python
"""jev_shadow_label.py — bulk Jev-shadow dataset collection (protocol §1, strategy 1).

Jev cloud labels synthetic-but-realistic domain states; we store Jev's SOFT distributions
as answers (soft labels train better — protocol §0) with source="jev-shadow".

Usage:
  laya-venv/bin/python scripts/jev_shadow_label.py --domain monitor-triage --n 1200
  laya-venv/bin/python scripts/jev_shadow_label.py --domain gmux-routing --n 500

Key: read from ~/.secrets/typesafe.env (TYPESAFE_API_KEY) — never hardcode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.typesafe.ai/v1/systemone"

# ---------------------------------------------------------------- question defs
SEV_CRIT = {
    "low": "cosmetic",
    "medium": "degradation",
    "high": "outage",
    "critical": "emergency",
}
CAUSE_CRIT = {
    "origin_down": "app server 5xx / process dead",
    "network": "connectivity, DNS or routing",
    "tls": "certificate problem",
    "upstream": "third-party dependency slow or failing",
}
INTENT_CRIT = {
    "lead": "commercial enquiry",
    "support": "existing user help",
    "spam": "junk",
}
TIER_CRIT = {
    "verified": "near-certain",
    "likely": "probably them",
    "weak": "coincidence possible",
    "junk": "false positive",
}
COMPLEX_CRIT = {
    "trivial": "one-liner",
    "simple": "mechanical single-file",
    "moderate": "multi-file reasoning",
    "complex": "architectural",
}
RISK_CRIT = {
    # CANONICAL gmux shape — must EXACTLY match crates/gmux-router routing_questions()
    # (labels + criteria strings). The 2026-09-23 audit caught a 3-option "local"
    # variant here that trained a DIFFERENT question than the router serves
    # (root cause of gmux-routing r1's calibration failure).
    "readonly": "no writes at all",
    "local-writes": "repo writes only, reversible",
    "destructive": "history rewrites, data loss, irreversible in-place changes",
    "external-side-effects": "touches infra, prod, external services or people",
}
Q_REASON = {
    "type": "noul",
    "instructions": "The task requires multi-step reasoning, not just a verdict",
}
Q_CONTEXT = {
    "type": "score",
    "instructions": "Context size needed",
    "criteria": ["small", "medium", "large"],
}
CLASS_CRIT = {
    "sensor": "IoT/environment sensor",
    "keyfob": "remote control or alarm fob",
    "voice": "voice or wideband comms",
    "noise": "noise or unclassified",
}

Q_SEVERITY = {"type": "choice", "instructions": "Severity", "criteria": SEV_CRIT}
Q_WAKE = {"type": "noul", "instructions": "Wake the owner at 3am"}
Q_CAUSE = {"type": "choice", "instructions": "Likely cause", "criteria": CAUSE_CRIT}
Q_INTENT = {"type": "choice", "instructions": "Contact intent", "criteria": INTENT_CRIT}
Q_TIER = {"type": "choice", "instructions": "Evidence tier", "criteria": TIER_CRIT}
Q_COMPLEX = {"type": "choice", "instructions": "Complexity", "criteria": COMPLEX_CRIT}
Q_RISK = {"type": "choice", "instructions": "Risk", "criteria": RISK_CRIT}
Q_CLASS = {
    "type": "choice",
    "instructions": "Transmission classification",
    "criteria": CLASS_CRIT,
}
Q_INTEREST = {"type": "noul", "instructions": "Worth flagging to the operator"}

# ---------------------------------------------------------------- state generators
SITES = [
    "endispute.com.au",
    "tradez.au",
    "qalarc.com",
    "qalarc.ai",
    "goetica.io",
    "hub.qalarc.dev",
    "api.tradez.au",
    "cdn.qalarc.ai",
    "monitor.qalarc.dev",
    "laya.qalarc.dev",
    "mail.endispute.com.au",
    "osint-hub.local",
    "coin.qalarc.dev",
    "status.qalarc.com",
    "forms.tradez.au",
]


def _site(rng):
    return rng.choice(SITES)


def gen_monitor(rng):
    """Monitor incident / email states. Returns (state, questions)."""
    kind = rng.choices(
        [
            "5xx",
            "latency",
            "tls",
            "dns",
            "green",
            "disk",
            "recovery",
            "contact",
            "flap",
            "timeout",
        ],
        weights=[18, 16, 12, 8, 12, 10, 8, 10, 3, 3],
    )[0]
    s = _site(rng)
    if kind == "5xx":
        code = rng.choice([500, 502, 503, 504])
        n = rng.randint(1, 12)
        dur = rng.choice(["2min", "5min", "12min", "34min", "1h", "3h"])
        extra = rng.choice(
            [
                "cert valid",
                "cert valid, origin pingable",
                "LB reports healthy backends",
                "origin CPU 95%",
                "origin oom-killed 20min ago",
                "after a deploy 15min ago",
                "",
            ]
        )
        extra = f", {extra}" if extra else ""
        st = f"{s} HTTP {code} {n} consecutive check{'s' if n > 1 else ''}, {dur}{extra}."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    elif kind == "latency":
        base = rng.choice([120, 180, 210, 260])
        mult = rng.uniform(1.2, 8.0)
        lat = int(base * mult)
        n = rng.randint(1, 6)
        region = rng.choice(
            ["from eu probe", "from all probes", "from us-east probe", "p95"]
        )
        st = f"{s} HTTP 200, latency {lat}ms (baseline {base}ms), {n} slow check{'s' if n > 1 else ''} {region}."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    elif kind == "tls":
        days = rng.choice([0, 1, 2, 3, 6, 10, 14, 21, 30, 45])
        green = "" if days <= 7 else ". All checks green"
        st = f"{s} TLS cert expires in {days} day{'s' if days != 1 else ''}{green}."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    elif kind == "dns":
        n = rng.randint(2, 9)
        st = (
            f"{s} DNS resolution failed on {n} of 12 checks (NXDOMAIN intermittent). "
            f"Resolver {rng.choice(['1.1.1.1', '8.8.8.8', 'systemd-resolved'])}."
        )
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    elif kind == "green":
        st = f"{s} all checks green, HTTP 200, latency {rng.randint(40, 300)}ms, cert {rng.randint(20, 300)} days left."
        qs = {"severity": Q_SEVERITY}
    elif kind == "disk":
        pct = rng.choice([78, 82, 86, 90, 92, 95, 97, 99])
        host = rng.choice(["origin", "db-primary", "monitor", "build", "hub"])
        st = f"{host} disk at {pct}% (grew {rng.randint(1, 9)}%/h last {rng.randint(1, 12)}h)."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    elif kind == "recovery":
        st = (
            f"{s} back to HTTP 200 after {rng.randint(4, 40)}min of 503s. "
            f"Latency {rng.randint(60, 400)}ms, error rate 0%. Recovered {rng.randint(1, 20)}min ago."
        )
        qs = {"severity": Q_SEVERITY}
    elif kind == "contact":
        texts = [
            "Hi, what are your consulting rates for a 2-week engagement?",
            "Do you offer bulk pricing for 10 sites?",
            "We need an OSINT integration, can you build it? Budget is flexible.",
            "URGENT!!! VIAGRA CHEAP >>> buy-now.example",
            "The dashboard breaks on Safari 16, any fix?",
            "Interested in a partnership, who should I talk to?",
            "Dear sir, I am a prince with a large inheritance...",
            "Can I get an invoice copy from last month?",
            "Quote for migrating 3 trucks telemetry by Friday?",
            "unsubscribe me please",
            "What is your SLA for the monitor tier?",
            "Crypto opportunity, guaranteed 40% returns, join now",
        ]
        st = f"Contact form: '{rng.choice(texts)}'"
        qs = {"intent": Q_INTENT}
    elif kind == "flap":
        st = f"{s} flapping: {rng.randint(3, 9)} up/down transitions in the last hour, currently UP."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE}
    else:
        st = f"{s} probe timeout after {rng.choice([5, 10, 15, 30])}s, {rng.randint(1, 4)} consecutive, site unreachable from {rng.choice(['one region', 'all regions'])}."
        qs = {"severity": Q_SEVERITY, "wake": Q_WAKE, "cause": Q_CAUSE}
    # occasionally drop a question to vary shapes
    if rng.random() < 0.12 and len(qs) > 1:
        qs.pop(rng.choice(list(qs.keys())))
    return st, qs


OSINT_SITES = [
    ("GitHub", "reliably 404s missing profiles", True),
    ("Reddit", "reliably 404s missing profiles", True),
    ("Pinterest", "returns 200 for missing profiles sometimes", False),
    ("Facebook", "returns a login wall regardless of profile existence", False),
    ("Mastodon", "reliably 404s missing profiles", True),
    ("Tumblr", "reliably 404s missing profiles", True),
    ("SoundCloud", "reliably 404s missing profiles", True),
    ("Vimeo", "reliably 404s missing profiles", True),
    ("Medium", "returns 200 with a 'stories not found' page sometimes", False),
    ("Flickr", "reliably 404s missing profiles", True),
    ("About.me", "reliably 404s missing profiles", True),
    ("DeviantArt", "returns 200 for deactivated profiles sometimes", False),
]
NAMES = [
    "johnsmith123",
    "m_keller",
    "dana.oreilly",
    "t4nk3r_88",
    "sarahconnor",
    "eli.wheeler",
    "quentin_t",
    "anna_b82",
    "ghost_rider",
    "maria.dlv",
    "jpkavanagh",
    "zoe.martin",
]


def gen_osint(rng):
    platform, behavior, reliable = rng.choice(OSINT_SITES)
    name = rng.choice(NAMES)
    code = (
        rng.choice([200, 200, 200, 404, 403])
        if reliable
        else rng.choice([200, 200, 200, 200, 403])
    )
    signals = []
    if code == 200:
        if rng.random() < 0.5:
            signals.append(f"display name '{name}' exact")
        if rng.random() < 0.4:
            signals.append("avatar matches reference photo (reverse-img hit)")
        if rng.random() < 0.35:
            signals.append(
                f"bio references {'the target city' if rng.random() < 0.5 else 'the target employer'}"
            )
        if rng.random() < 0.3:
            signals.append("account active this week")
        if rng.random() < 0.25:
            signals.append("joined recently (this year), no followers")
        if rng.random() < 0.2:
            signals.append("profile empty: default avatar, zero posts")
    st = (
        f"site={platform} url=https://{'github.com' if platform == 'GitHub' else platform.lower() + '.com'}/{name} "
        f"status={code}. {signals and '; '.join(signals) + '. ' or ''}Platform {behavior}."
    )
    return st, {"tier": Q_TIER}


def gen_gmux(rng):
    kind = rng.choices(
        ["trivial", "simple", "moderate", "complex"], weights=[25, 35, 25, 15]
    )[0]
    risk_kind = rng.choices(["readonly", "local", "destructive"], weights=[30, 55, 15])[
        0
    ]
    task = {
        "trivial": rng.choice(
            [
                "fix typo in README",
                "bump version to 1.2.3",
                "rename variable `x` to `count`",
                "add missing semicolon",
                "update copyright year",
                "fix broken link in docs",
                "change log level INFO to DEBUG",
                "delete commented-out line",
            ]
        ),
        "simple": rng.choice(
            [
                "add a CLI flag that toggles an existing feature",
                "write unit tests for utils.py",
                "convert this module to type hints",
                "extract duplicated JSON parsing into a helper",
                "add retry with backoff to the HTTP client",
                "update dependencies to latest patch",
            ]
        ),
        "moderate": rng.choice(
            [
                "add pagination to the API and its client, update tests",
                "migrate config parsing from JSON to TOML across services",
                "implement rate limiting middleware with tests",
                "refactor the DB layer to async, keep the public API",
                "add a webhook subsystem with signing and retries",
            ]
        ),
        "complex": rng.choice(
            [
                "refactor the bus module across 3 crates, async patterns, update tests",
                "redesign the scheduler for multi-tenant fairness (architectural)",
                "split the monolith into workspace crates with a stable internal API",
                "introduce event sourcing for the state store",
                "rework auth to support OIDC + SCIM across all services",
            ]
        ),
    }[kind]
    risk_task = {
        "readonly": rng.choice(
            [
                "analyze the repo and report findings",
                "grep for deprecated calls, list them",
                "estimate migration effort, no changes",
                "review the diff and comment",
                "profile the test suite and report slow tests",
            ]
        ),
        "local": rng.choice(
            [
                "commit to a feature branch",
                "run the formatter and fix lint across src/",
                "update tests to match the new API",
                "write the missing module docstrings",
                "apply the patch from the issue thread",
            ]
        ),
        "destructive": rng.choice(
            [
                "force-push the cleaned history to main",
                "drop and recreate the prod database schema",
                "rotate and revoke all API keys now",
                "migrate prod data in place (irreversible)",
                "terraform destroy the staging stack",
            ]
        ),
    }[risk_kind]
    st = (
        f"Task: {task}. {risk_task[0].upper()}{risk_task[1:]}."
        if not task.endswith(".")
        else f"Task: {task} {risk_task}."
    )
    return st, {
        "complexity": Q_COMPLEX,
        "risk": Q_RISK,
        "needs_reasoning": Q_REASON,
        "context_size": Q_CONTEXT,
    }


def gen_rfai(rng):
    kind = rng.choices(
        ["sensor", "keyfob", "voice", "noise"], weights=[40, 25, 15, 20]
    )[0]
    dbm = rng.randint(-95, -40)
    if kind == "sensor":
        st = (
            f"RF transmission cluster: {rng.choice(['433MHz', '868MHz', '315MHz'])}, "
            f"burst pattern {rng.randint(2, 5)}x short + pause, consistent with cheap "
            f"{rng.choice(['temperature', 'humidity', 'door contact'])} sensor, power {dbm}dBm, "
            f"recurring {rng.choice([17, 30, 47, 60, 120])}s interval."
        )
    elif kind == "keyfob":
        st = (
            f"RF burst: {rng.choice(['433MHz', '315MHz', '868MHz'])} alternating long-short pattern "
            f"typical of {'rolling-code keyfob' if rng.random() < 0.7 else 'garage remote'}, "
            f"{dbm}dBm ({'close' if dbm > -60 else 'distant'}), "
            f"{'single occurrence' if rng.random() < 0.5 else f'{rng.randint(2, 6)} presses over 2min'}."
        )
    elif kind == "voice":
        st = (
            f"RF: wideband {rng.choice(['analog FM voice', 'digital voice (DMR)', 'analog AM'])} "
            f"activity on {rng.choice(['146MHz', '446MHz PMR', '27MHz CB'])}, {dbm}dBm, "
            f"intermittent {rng.randint(5, 90)}s transmissions with speech cadence."
        )
    else:
        st = (
            f"RF: broadband hash across {rng.choice(['300-500MHz', '800-900MHz'])}, no stable pattern, "
            f"{dbm}dBm, spikes correlate with {rng.choice(['microwave oven cycles', 'nearby switching PSU', 'vehicle ignition'])}. "
            f"No decodable frames."
        )
    return st, {"class": Q_CLASS, "interest": Q_INTEREST}


# ---------------------------------------------------------------- GEO domains (2026-09-22)
# States mirror the EXACT vocabulary of qalarc.ai/scripts/geo_pipeline.py check_site()
# output + the SEO_strategy backlog/04 finding types, so trained Laya triages real audits.

GEO_SITES = [
    ("qalarc.com", 10),
    ("chanalyse.org", 8),
    ("tradez.au", 7),
    ("goetica.ai", 6),
    ("volkus.net", 6),
    ("doof.ing", 5),
    ("gmux.ai", 4),
    ("x10.au", 4),
    ("endispute.com.au", 4),
    ("osint.qalarc.com", 3),
    ("museall.qalarc.com", 3),
    ("biz.sydney", 3),
]

Q_PRIORITY = {
    "type": "choice",
    "instructions": "Fix priority for this GEO audit finding",
    "criteria": {
        "p0": "blocks indexing or domain dead - fix infrastructure before any SEO",
        "p1": "high value - costs citations or visibility right now",
        "p2": "worthwhile but not urgent",
        "skip": "cosmetic or no measurable effect",
    },
}
Q_FIXNOW = {"type": "noul", "instructions": "Fix in the current deploy window"}
Q_ATTR = {
    "type": "choice",
    "instructions": "Attribution class of this AI-engine answer about a Qalarc product",
    "criteria": {
        "correct": "attributes the work to Qalarc and the facts check out",
        "wrong-person": "credits a person or organization that is not Qalarc",
        "wrong-facts": "Qalarc is named but the facts are wrong, stale or confused",
        "no-citation": "describes the product with no source or attribution at all",
    },
}
Q_ESC = {"type": "noul", "instructions": "Escalate to the owner for correction action"}

GEO_ISSUES = [
    "CRITICAL: homepage HTTP {code}",
    "title weak/too short: '{frag}'",
    "no meta description",
    "no valid JSON-LD",
    "llms.txt missing ({code})",
    "robots.txt has no explicit AI-bot rules (implicit allow, but declare them)",
    "sitemap.xml missing ~90 sub-pages under /projects/qals/",
    "llms.txt org facts say 79 documented projects but projects.json has 86 (stale facts)",
    "panel entry missing updatedAt (invisible in the Latest showcase)",
]
GEO_FRAGS = ["Home", "Welcome", "Untitled", "qalarc", "index", "Site"]


def gen_geo_audit(rng):
    domain, w = rng.choices(GEO_SITES, weights=[x[1] for x in GEO_SITES])[0]
    kind = rng.choices(["clean", "one", "pile", "dead"], weights=[30, 35, 25, 10])[0]
    if kind == "dead":
        issues = [f"CRITICAL: homepage HTTP {rng.choice([500, 502, 503, 504])}"]
        if rng.random() < 0.6:
            issues.append(f"llms.txt missing ({rng.choice([404, 503])})")
    elif kind == "clean":
        issues = []
    elif kind == "one":
        issues = [rng.choice(GEO_ISSUES[1:])]
    else:
        k = rng.randint(2, 4)
        issues = rng.sample(GEO_ISSUES[1:], k)
    iss = "; ".join(
        i.format(code=rng.choice([404, 500, 503]), frag=rng.choice(GEO_FRAGS))
        for i in issues
    )
    clean = "all checks clean - llms.txt 200, AI bots declared, JSON-LD valid, sitemap current"
    st = f"GEO audit {domain} (weight {w}): {iss or clean}."
    return st, {"priority": Q_PRIORITY, "fixnow": Q_FIXNOW}


GEO_PRODUCTS = [
    (
        "Qalarc Token Killer (QTK)",
        "token compression plugin for AI coding agents, sibling of RTK",
    ),
    ("NetSleuth", "passive network surveillance tool"),
    ("chanalyse", "4chan discourse-intelligence engine"),
    ("goetica", "character-truth workshop / AI character platform"),
    ("Volkus", "demographic-transparent community experiment"),
    ("Herald", "stroke communicator for severe aphasia"),
    ("Doof / doof.ing", "Phone Link AI across Signal, WhatsApp and web"),
    ("tradez.au", "tradie directory with auto-generated sites"),
]
GEO_ENGINES = [
    "Brave Leo",
    "ChatGPT Search",
    "Perplexity",
    "Google AI Overviews",
    "Copilot",
    "Claude with web search",
]
GEO_QUERIES = [
    '"{prod}" - which company built it?',
    "who makes {prod}?",
    "what is {prod} and who is behind it?",
    "{prod} author and source repository?",
]


def gen_geo_attr(rng):
    prod, desc = rng.choice(GEO_PRODUCTS)
    engine = rng.choice(GEO_ENGINES)
    q = rng.choice(GEO_QUERIES).format(prod=prod)
    variant = rng.choices(
        ["correct", "wrong-person", "wrong-facts", "no-citation"],
        weights=[38, 22, 25, 15],
    )[0]
    if variant == "correct":
        cite = rng.choice(
            [
                f"cites github.com/qalarc and qalarc.com",
                f"cites qalarc.com project page as the source",
                f"says 'built by Qalarc' and links the repo",
            ]
        )
        extra = rng.choice(
            [
                f"describes it as a {desc}",
                f"describes it as a {desc}; facts match the live page",
                f"one stat slightly out of date but attribution and category right",
            ]
        )
        st = f'{engine} on {q} -> answer: "{prod} is built by Qalarc, {desc}. {cite}." {extra}.'
    elif variant == "wrong-person":
        who = rng.choice(
            [
                "John Olten",
                "an anonymous 4chan poster",
                "a crypto DAO",
                "Ruskov Systems",
                "an unaffiliated researcher",
            ]
        )
        st = (
            f'{engine} on {q} -> answer: "{prod} was created by {who}" '
            f"with no Qalarc mention and no repo link."
        )
    elif variant == "wrong-facts":
        wrong = rng.choice(
            [
                f"calls it a cryptocurrency token with price predictions",
                f"confuses it with RTK and says it is a fork by a different vendor",
                f"says it is closed-source enterprise SaaS",
                f"claims 54k GitHub stars (actual is 80k+) and 13 supported agents (actual 14)",
            ]
        )
        st = f'{engine} on {q} -> answer: "{prod} is by Qalarc, {wrong}."'
    else:
        st = (
            f"{engine} on {q} -> answer describes a {desc} generically, "
            f"no maker named, no citation given."
        )
    return st, {"attribution": Q_ATTR, "escalate": Q_ESC}


GENERATORS = {
    "monitor-triage": gen_monitor,
    "osint-grading": gen_osint,
    "gmux-routing": gen_gmux,
    "rfai-feed": gen_rfai,
    "geo-audit-triage": gen_geo_audit,
    "geo-attribution": gen_geo_attr,
}


# ---------------------------------------------------------------- seo-grading (§4.8)
SEO_SITES = [
    (
        "qalarc.com",
        ["qalarc AI decision stack", "local Jev fine-tune", "qalarc OSINT tools"],
    ),
    (
        "tradez.au",
        ["tradie marketplace Australia", "trade quotes online", "find a tradie"],
    ),
    (
        "endispute.com.au",
        ["dispute resolution Australia", "mediation services", "NDIS disputes"],
    ),
    ("doof.ing", ["bush doof library", "doof prophecies", "festival archive"]),
]
SEO_TEMPLATES = [
    "guide",
    "product page",
    "blog post",
    "landing page",
    "docs page",
    "case study",
]


def gen_seo(rng):
    """SEO grading states (TRAINING_AREAS §4.8): page snapshot vs target query."""
    site, queries = rng.choice(SEO_SITES)
    q = rng.choice(queries)
    path = rng.choice(
        [
            "/guides/x",
            "/services/y",
            "/blog/post-slug",
            "/",
            "/docs/topic",
            "/case-studies/z",
        ]
    )
    template = rng.choice(SEO_TEMPLATES)
    words = rng.choice([80, 150, 320, 600, 900, 1400, 2200])
    title_len = rng.randint(18, 90)
    has_meta = rng.random() < 0.75
    h2s = rng.randint(0, 9)
    updated = rng.choice(
        ["3 days ago", "2 months ago", "11 months ago", "4 years ago", "last week"]
    )
    pos = rng.choice([1, 3, 7, 12, 28, 45, 88])
    sibling = rng.random() < 0.5
    st = (
        f"page={site}{path} template={template} target_query='{q}' | title {title_len} chars"
        f"{', meta present' if has_meta else ', META MISSING'} | {words} words, {h2s} h2 sections, "
        f"updated {updated} | current position #{pos}"
        + (
            f" | sibling page '{site}/related' targets a near-identical query"
            if sibling
            else ""
        )
    )
    qs = {
        "page_quality": {
            "type": "score",
            "instructions": "Overall page quality for the target query",
            "criteria": [
                "thin/stub",
                "sparse",
                "adequate",
                "strong",
                "comprehensive & fresh",
            ],
        },
        "title_meta": {
            "type": "score",
            "instructions": "Title + meta vs the query",
            "criteria": ["missing/poor", "weak", "decent", "optimized"],
        },
        "intent_match": {
            "type": "choice",
            "instructions": "Page vs query intent",
            "criteria": {
                "matches": "serves the query intent",
                "partial": "partially serves",
                "mismatched": "wrong intent",
            },
        },
        "cannibalization": {
            "type": "choice",
            "instructions": "Cannibalization risk",
            "criteria": {
                "none": "no sibling conflict",
                "possible": "overlap possible",
                "likely": "sibling competes for this query",
            },
        },
        "change_significant": {
            "type": "noul",
            "instructions": "A content change here would be meaningful (not template churn)",
        },
    }
    if rng.random() < 0.15:
        qs.pop(rng.choice(list(qs.keys())))
    return st, qs


# ---------------------------------------------------------------- compliance-marks (Doof.ing library)
LIB_CATEGORIES = [
    "core-beliefs",
    "sacred-practices",
    "prophecies",
    "manifesto",
    "revelations",
    "sacred-spaces",
]
LIB_ON_THEME = [
    "the doof floor becomes a living mandala at peak bass",
    "sparklers are the only permitted candles in the temple of sound",
    "every beat dropped in the forest echoes in the dreamtime",
    "hydration is communion; the water point is our baptismal font",
    "the Generator is a false idol; silence between sets is sacred",
    "sunrise sets align the collective third eye of the crowd",
]
LIB_OFF_THEME = [
    "buy 5000 instagram followers cheap, link in bio",
    "corporate team-building seminars for mid-size enterprises",
    "HOW TO MAKE $$$ FROM HOME — CLICK NOW",
    "quarterly compliance training module 7: workplace ergonomics",
    "the stock ticker at 3am is a spiritual experience",
    "my cryptocurrency podcast episode 214",
]
LIB_VARIANT_ANGLES = [
    "reframed as a cosmic law",
    "given a numbers-mysticism twist",
    "tied to the sacred weekend forecast",
    "phrased as an ancient prophecy fulfilled",
    "stated as community bylaws",
    "delivered as a DJ's closing benediction",
]


def gen_library_marks(rng):
    """Compliance marks for the Doof.ing library (catalog + community votes).
    Decide keep/canonical/on-theme for candidate entries."""
    kind = rng.choices(["catalog", "on-theme", "off-theme"], weights=[30, 45, 25])[0]
    votes = rng.randint(0, 60)
    if kind == "catalog":
        st = (
            f"library entry: 'the bass will drop and dimensions will align' [prophecies] "
            f"interpretation: sound frequencies create dimensional alignment. community votes: {votes}."
        )
    elif kind == "on-theme":
        text = rng.choice(LIB_ON_THEME)
        angle = rng.choice(LIB_VARIANT_ANGLES)
        cat = rng.choice(LIB_CATEGORIES)
        st = (
            f"library candidate: '{text}' [{cat}], {angle}. "
            f"interpretation: consistent with the established doof canon. community votes: {votes}."
        )
    else:
        st = (
            f"library candidate: '{rng.choice(LIB_OFF_THEME)}' [misc], submitted via open form. "
            f"interpretation: unclear relation to the canon. community votes: {votes}."
        )
    qs = {
        "keep": {
            "type": "choice",
            "instructions": "Library compliance decision",
            "criteria": {
                "keep": "fits the library's purpose and standards",
                "review": "needs a human look",
                "remove": "violates scope/spam",
            },
        },
        "canonical": {
            "type": "noul",
            "instructions": "This entry is core canon of the library",
        },
        "on_topic": {
            "type": "choice",
            "instructions": "Theme check",
            "criteria": {
                "on-theme": "matches the library's theme",
                "tangential": "loosely related",
                "off-theme": "unrelated or spam",
            },
        },
    }
    if rng.random() < 0.15:
        qs.pop(rng.choice(list(qs.keys())))
    return st, qs


GENERATORS["seo-grading"] = gen_seo
GENERATORS["compliance-marks"] = gen_library_marks


# ---------------------------------------------------------------- jev client
def jev_call(state, questions, model="jev-1.13.0", retries=4):
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    delay = 1.0
    for attempt in range(retries):
        req = urllib.request.Request(
            API,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def to_soft_answers(questions, resp_answers):
    """Store Jev's SOFT distributions (protocol §0: soft labels train better).
    choice/score -> {"probabilities": {...}}; noul -> float in [0,1]."""
    out = {}
    for qid in questions:
        a = resp_answers.get(qid)
        if not a:
            continue
        if a.get("type") == "noul":
            out[qid] = round(float(a["noul"]), 4)
        elif "probabilities" in a:
            out[qid] = {
                "probabilities": {
                    k: round(float(v), 4) for k, v in a["probabilities"].items()
                }
            }
        elif "choice" in a:
            out[qid] = a["choice"]
    return out


def label_one(args):
    seed, domain = args
    rng = random.Random(seed)
    state, questions = GENERATORS[domain](rng)
    resp = jev_call(state, questions)
    answers = to_soft_answers(questions, (resp or {}).get("answers", {}))
    rec = {
        "state": state,
        "questions": questions,
        "answers": answers,
        "source": "jev-shadow",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=sorted(GENERATORS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--seed0", type=int, default=None, help="first rng seed (unique per batch)"
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    env = Path.home() / ".secrets/typesafe.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY missing")

    out = Path(a.out or ROOT / "datasets" / f"{a.domain}.jsonl")
    seed0 = a.seed0 if a.seed0 is not None else int(time.time()) % 10_000_000
    jobs = [(seed0 + i, a.domain) for i in range(a.n)]

    ok = fail = 0
    t0 = time.time()
    with out.open("a") as f:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(label_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rec = fut.result()
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(
                        f"[{i}/{a.n}] FAIL {futs[fut]}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                if i % 50 == 0 or i == a.n:
                    rate = i / (time.time() - t0)
                    print(
                        f"[{i}/{a.n}] ok={ok} fail={fail} {rate:.1f} rows/s eta {(a.n - i) / max(rate, 0.01):.0f}s",
                        flush=True,
                    )
    print(f"DONE {a.domain}: +{ok} rows (fail={fail}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
