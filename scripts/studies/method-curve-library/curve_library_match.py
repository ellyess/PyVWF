"""The brand-and-spec matcher for the curve library study's T1 condition.

T1 asks what happens when a unit is simulated on the curve of the machine the
register says it is, rather than on the nearest specific power. That needs the
register's own designation mapped to a licensed-library key, and the two do not
share a vocabulary: a Danish row reads ``Vestas Wind Systems A/S`` and
``V 47-660`` where the library reads ``Vestas.V47.660``; a British row packs
both into one field, ``Vestas V90 3000``; an American row reads
``GE Wind GE1.5-77``.

**Exact normalised match only, and the normalisation is the whole design.**
Both sides are reduced to upper-case alphanumerics with every separator
removed, so ``V 47-660`` and ``V47.660`` become ``V47660``. A manufacturer
alias table maps each register's spelling of a maker onto the library's. A key
matches if the register text equals the key's normalised form with its rating,
or without it, since some registers name the machine and not its rating.

What this deliberately does NOT do: no fuzzy matching, no nearest specific
power, no inference from rating and diameter. Those are what T0 already does,
and a matcher that guessed would make T1 a second specific-power match under
another name. A unit this cannot place keeps its T0 assignment, and the share
it can place is T1's coverage, reported per region and gating the region in
`docs/findings/method-curve-library-prereg.md`.

Read-only, and importable: the study driver and the tests use the same rules.
"""

import re
import unicodedata

#: Register spellings of a manufacturer, mapped onto the library's spelling.
#: Fixed before any T1 run. A maker absent here is normalised as written, which
#: matches when the two already agree (``Nordex``, ``Enercon``, ``Gamesa``).
MANUFACTURER_ALIASES = {
    "vestaswindsystemsas": "vestas",
    "vestaswindsystems": "vestas",
    "negmicon": "negmicon",
    "negmiconas": "negmicon",
    "siemenswindpower": "siemens",
    "siemensgamesarenewableenergy": "siemensgamesa",
    "solidwindpoweras": "solidwindpower",
    "gaiawindltd": "gaiawind",
    "gewind": "ge",
    "geenergy": "ge",
    "generalelectric": "ge",
    "repowersystems": "repower",
    "bonusenergy": "bonus",
    "bonusenergyas": "bonus",
    "nordexenergy": "nordex",
    "enerconge": "enercon",
    "enerconggmbh": "enercon",
}

#: Register values that name no machine. Treated as unmatchable, not as a name.
UNKNOWN = {"", "unknown", "na", "none", "unspecified", "other", "nan"}


def normalise(text: object) -> str:
    """Upper-case alphanumerics only, accents folded, everything else dropped."""
    if text is None:
        return ""
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-zA-Z]", "", folded).lower()


def canonical_manufacturer(text: object) -> str:
    """A register's spelling of a maker, as the library spells it."""
    key = normalise(text)
    return MANUFACTURER_ALIASES.get(key, key)


def key_forms(model_key: str) -> tuple[str, str]:
    """A library key's normalised form with its rating, and without it.

    ``Vestas.V47.660`` gives ``vestasv47660`` and ``vestasv47``. The rating is
    the last dot-separated field when it is a bare number: ``REpower.3.4M`` and
    ``GE.1.5sle`` have none, and both forms are then the same.
    """
    parts = str(model_key).split(".")
    full = normalise(model_key)
    if len(parts) > 2 and parts[-1].isdigit():
        return full, normalise(".".join(parts[:-1]))
    return full, full


def build_index(models) -> dict[str, str]:
    """Normalised form to library key, for every key in ``models``.

    ``models`` is a frame with ``manufacturer`` and ``model`` columns. A form
    two keys share is dropped rather than resolved: an ambiguous match is not
    an exact one.
    """
    index: dict[str, str] = {}
    clashes: set[str] = set()
    for maker, key in zip(models["manufacturer"], models["model"]):
        canonical = canonical_manufacturer(maker)
        for form in set(key_forms(key)):
            for candidate in {form, canonical + form, canonical + form.removeprefix(canonical)}:
                if not candidate:
                    continue
                if index.get(candidate, key) != key:
                    clashes.add(candidate)
                index[candidate] = key
    for form in clashes:
        index.pop(form, None)
    return index


#: A rating written at the end of a designation, with the separator before it:
#: "V110-2.0", "SWT 2.3-93" (no, that is a diameter), "Gaia Wind 133-10 kW".
#: Matched on the RAW text, because normalising first glues the rating to the
#: model number and the two can no longer be told apart ("V110-2.0" becomes
#: "v11020", from which nothing can be recovered).
TRAILING_RATING = re.compile(
    r"[\s\-_/]+(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mw|kw)?\s*$", re.IGNORECASE
)
#: Below this, an unlabelled trailing rating is read as megawatts and above it
#: as kilowatts. No machine in these registers is rated between 100 kW and
#: 100 MW, so the split is unambiguous for the data, and a register that writes
#: its unit ("10 kW") is believed rather than guessed at.
MW_BELOW = 100


def without_trailing_rating(text: object) -> str:
    """A raw designation with a trailing rating dropped, normalised.

    The library writes a rating in kW and the registers do not agree with it:
    ``Vestas V90 3000`` matches ``Vestas.V90.3000`` outright, while
    ``Vestas V110-2.0`` says the same machine in MW and matches nothing. Both
    sides therefore also compare with the rating removed, which is what
    ``key_forms`` already does to a key.

    It is a second candidate, never a replacement: the form with the rating is
    tried too, and a designation that means two different machines without its
    rating is dropped as ambiguous by ``build_index``.
    """
    raw = str(text) if text is not None else ""
    stripped = TRAILING_RATING.sub("", raw, count=1)
    return normalise(stripped) if normalise(stripped) else normalise(raw)


def rating_in_kilowatts(text: object) -> str:
    """A designation with its trailing rating restated in kilowatts.

    ``V80-2.0`` and ``Vestas.V80.2000`` are one machine, and the library keeps
    both ``V80.1800`` and ``V80.2000``, so dropping the rating leaves an
    ambiguity that only the rating resolves. Converting it is exact: the
    register states a number and a unit, written or implied by ``MW_BELOW``,
    and this restates it. It does not infer a rating that is not there.

    Returns "" when the designation ends in no rating.
    """
    raw = str(text) if text is not None else ""
    found = TRAILING_RATING.search(raw)
    if not found:
        return ""
    value = float(found.group("value").replace(",", "."))
    unit = (found.group("unit") or "").lower()
    kw = value * 1000 if unit == "mw" or (not unit and value < MW_BELOW) else value
    if kw != int(kw):
        return ""
    return normalise(raw[: found.start()]) + str(int(kw))


def register_forms(manufacturer: object, model: object) -> set[str]:
    """Every normalised form a register row could be written as.

    A register may hold the maker and the machine in separate fields (Denmark),
    the two packed into one (the United Kingdom), or the maker repeated inside
    the machine's name (``GE Wind GE1.5-77``). All three reduce to the same
    candidates, and a row whose fields say nothing yields none.
    """
    maker_raw, model_raw = normalise(manufacturer), normalise(model)
    if maker_raw in UNKNOWN and model_raw in UNKNOWN:
        return set()
    maker = canonical_manufacturer(manufacturer) if maker_raw not in UNKNOWN else ""
    machine = model_raw if model_raw not in UNKNOWN else ""
    bare = without_trailing_rating(model) if machine else ""
    in_kw = rating_in_kilowatts(model) if machine else ""
    forms = {maker + machine, machine, maker, maker + bare, bare, maker + in_kw, in_kw}
    if machine.startswith(maker) and maker:
        forms.add(machine)  # the maker repeated in the name
        forms.add(bare)
    return {f for f in forms if f and f not in UNKNOWN}


def match(manufacturer: object, model: object, index: dict[str, str]) -> str | None:
    """The library key this register row names exactly, or None.

    A row matching two keys through different forms is unmatched: exact means
    unambiguous. The maker alone never matches a key, since that is a brand and
    not a machine.
    """
    forms = register_forms(manufacturer, model)
    maker = canonical_manufacturer(manufacturer)
    hits = {index[f] for f in forms if f in index and f != maker}
    return hits.pop() if len(hits) == 1 else None
