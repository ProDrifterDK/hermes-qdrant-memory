"""Heuristic detector for durable statements in a person's own messages (capture lane A).

Detection only: it says "this message probably states something that holds beyond the
session, and of what kind". It does not rewrite the message into a standalone fact, so its
hits only ever feed the review queue (design I7: without a second signal, heuristic hits may
only enter review).

The rules are generic Spanish and English signal forms: asking to remember or record
something, never/always, stated preferences, corrections, "from now on", scope decisions,
"must / must not". They carry no confidence and no threshold: a hit is a hit. They run only
on rows the authorship classifier marked ``human`` (capture_reader), on the body with
envelopes stripped, after code, quoted lines and links are removed.

Known limits: it cannot resolve a bare approval ("dale", "ok, the second one") into a fact,
it misses implicit facts, and quoted or pasted text that keeps these forms fires.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Word boundaries that also hold around Spanish accented letters.
_W = r"(?<![\wáéíóúñü])"
_E = r"(?![\wáéíóúñü])"

# (rule id, kind, pattern), in priority order: the first matching rule names the kind.
USER_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("told_you", "correction", re.compile(
        _W + r"(te dije que|ya te dije|como te dije|te lo dije|i told you|as i said|eso rompe|eso está mal|"
        r"eso esta mal|that'?s wrong|that is wrong)" + _E, re.I)),
    ("initial_no", "correction", re.compile(r"^\s*no[.,!]\s+\S", re.I)),
    ("remember", "invariant", re.compile(
        _W + r"(recuerda|record[aá]|acu[eé]rdate|no olvides|ten en cuenta|ten[eé] en cuenta|remember|keep in mind|"
        r"don'?t forget)" + _E, re.I)),
    ("from_now_on", "invariant", re.compile(
        _W + r"(de ahora en adelante|a partir de ahora|desde ahora|en adelante|from now on|going forward|"
        r"por defecto|by default)" + _E, re.I)),
    ("never_always", "invariant", re.compile(_W + r"(nunca|jam[aá]s|siempre|never|always)" + _E, re.I)),
    ("all_we_do", "invariant", re.compile(
        _W + r"tod[oa]s? (?:l[oa]s )?\w+ (?:l[oa]s )?(?:hacemos|usamos|corremos|van|corren|ejecutamos)" + _E, re.I)),
    ("must_general", "invariant", re.compile(
        _W + r"(debe(?:n)? ser|no debe(?:n|s)?|no deber[ií]a|must not|must always|should never|has to be)" + _E,
        re.I)),
    ("prefer", "preference", re.compile(
        _W + r"(prefiero|preferir[ií]a|me gusta(?:r[ií]a)? que|no me gusta|no quiero que|i prefer|i'd rather|"
        r"i would rather|i don'?t want you to|i don'?t like)" + _E, re.I)),
    ("record_request", "preference", re.compile(
        _W + r"(guarda(?:lo)?|guard[aá]|registra(?:lo)?|registr[aá]|anota(?:lo)?|deja(?:lo)? registrado|"
        r"dej[aá] registrado|save (?:this|that|it)|store (?:this|that)|note (?:this|that) down|"
        r"en tus (?:skills|memorias?)|en (?:tu|la) memoria|in your memory)" + _E, re.I)),
    ("scope", "decision", re.compile(_W + r"(fuera del scope|fuera de(?:l)? alcance|out of scope)" + _E, re.I)),
    ("we_have", "fact", re.compile(
        _W + r"(ya tenemos|tenemos otr[oa]|we have another|we already have)" + _E, re.I)),
)

_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"\b(?:https?|file)://\S+|@url:\S+")
_QUOTED_LINE = re.compile(r"^\s*>.*$", re.M)


@dataclass(frozen=True)
class Detection:
    kind: str
    rule_ids: tuple[str, ...]


def detection_text(body: str) -> str:
    """The body with code, quoted lines and links removed: what the rules look at."""
    text = _CODE_BLOCK.sub(" ", str(body or ""))
    text = _QUOTED_LINE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    return _URL.sub(" ", text)


def detect(body: str) -> Detection | None:
    """The rules that fire on a person's message body, or ``None``."""
    text = detection_text(body)
    if not text.strip():
        return None
    hits = [(rule_id, kind) for rule_id, kind, pattern in USER_RULES if pattern.search(text)]
    if not hits:
        return None
    return Detection(kind=hits[0][1], rule_ids=tuple(rule_id for rule_id, _ in hits))


# --- what may never auto-apply (design I9) ----------------------------------------------------

_SENSITIVE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("identity", re.compile(
        _W + r"(my|mi|mis) (name|surname|email|e-mail|phone|address|birthday|birth date|age|passport|"
        r"nombre|apellido|correo|tel[eé]fono|celular|direcci[oó]n|cumplea[nñ]os|edad|pasaporte|rut|dni)" + _E, re.I)),
    ("money", re.compile(
        _W + r"(usd|eur|clp|\$\s?\d|dinero|money|precio|price|pago|payment|invertir|invest\w*|inversi[oó]n|"
        r"sueldo|salary|salario|tarjeta|card|banco|bank|cr[eé]dito|credit)" + _E, re.I)),
    ("security", re.compile(
        _W + r"(password|contrase[nñ]a|clave|token|api[ _-]?key|secret|secreto|ssh|credential\w*|credencial\w*|"
        r"firewall|2fa|mfa|security|seguridad|permiso\w*|permission\w*|sudo|root)" + _E, re.I)),
)


def sensitive_flags(text: str) -> tuple[str, ...]:
    """Identity, money and security markers: such items never auto-apply."""
    return tuple(name for name, pattern in _SENSITIVE if pattern.search(str(text or "")))
