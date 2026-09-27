"""Model portfolio files: schema, loading and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CASH_SLEEVE = "cash"
CASH_INSTRUMENTS = frozenset({"USD", "USDC"})
INSTRUMENT_KEYS = ("regular", "extended", "overnight", "weekend", "any")

# Which instrument-map keys apply in each market session, in order. The first key a sleeve
# defines wins, even when its list is empty: `weekend: []` means frozen, not "fall back to any".
# Pre/post fall back to the overnight list because both are thin, limit-only sessions.
SESSION_LOOKUP = {
    "regular": ("regular", "any"),
    "pre": ("extended", "overnight", "any"),
    "post": ("extended", "overnight", "any"),
    "overnight": ("overnight", "any"),
    "pause": ("weekend", "any"),
    "weekend": ("weekend", "any"),
    "closed": ("weekend", "any"),
}

_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TOL = 1e-9


class ModelError(ValueError):
    def __init__(self, source: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{source}: " + "; ".join(problems))


@dataclass(frozen=True)
class Band:
    abs: float | None = None
    rel: float | None = None

    def width(self, target: float) -> float:
        """Allowed drift either side of target. A sleeve is out as soon as it leaves either band."""
        widths = []
        if self.abs is not None:
            widths.append(self.abs)
        if self.rel is not None:
            widths.append(self.rel * target)
        return min(widths) if widths else 0.0


@dataclass(frozen=True)
class Sleeve:
    id: str
    target: float
    band: Band | None
    instruments: dict[str, tuple[str, ...]]
    max_off_hours_pct: float = 0.25

    @property
    def is_cash(self) -> bool:
        return self.id == CASH_SLEEVE

    @property
    def all_instruments(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for names in self.instruments.values():
            for name in names:
                seen.setdefault(name, None)
        return tuple(seen)

    def instruments_for(self, session: str) -> tuple[str, ...]:
        for key in SESSION_LOOKUP[session]:
            if key in self.instruments:
                return self.instruments[key]
        return ()

    def band_edges(self) -> tuple[float, float] | None:
        if self.band is None:
            return None
        width = self.band.width(self.target)
        return max(self.target - width, 0.0), self.target + width


@dataclass(frozen=True)
class Model:
    name: str
    version: int
    base_currency: str
    cash_floor: float
    sleeves: tuple[Sleeve, ...]
    source: str = field(default="", compare=False)

    @property
    def cash(self) -> Sleeve:
        return next(s for s in self.sleeves if s.is_cash)

    @property
    def invested(self) -> tuple[Sleeve, ...]:
        return tuple(s for s in self.sleeves if not s.is_cash)

    def sleeve(self, sleeve_id: str) -> Sleeve:
        return next(s for s in self.sleeves if s.id == sleeve_id)

    def sleeve_for_instrument(self, instrument: str) -> Sleeve:
        return next(s for s in self.sleeves if instrument in s.all_instruments)


def load_model(path: str | Path) -> Model:
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return parse_model(raw, source=str(path))


def load_models(directory: str | Path) -> dict[str, Model]:
    models = [load_model(p) for p in sorted(Path(directory).glob("*.yaml"))]
    by_name: dict[str, Model] = {}
    for model in models:
        if model.name in by_name:
            raise ModelError(
                model.source,
                [f"model name {model.name!r} also used in {by_name[model.name].source}"],
            )
        by_name[model.name] = model
    return by_name


def parse_model(raw: object, source: str = "<model>") -> Model:
    problems: list[str] = []
    if not isinstance(raw, dict):
        raise ModelError(source, ["top level must be a mapping"])

    unknown = set(raw) - {
        "model",
        "version",
        "base_currency",
        "cash_floor",
        "sleeves",
        "description",
    }
    if unknown:
        problems.append(f"unknown keys: {sorted(unknown)}")

    name = raw.get("model")
    if not isinstance(name, str) or not _NAME.match(name):
        problems.append("model must be a lowercase slug like 'growth-247'")

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        problems.append("version must be a positive integer")

    base_currency = raw.get("base_currency", "USD")
    if base_currency != "USD":
        problems.append("only USD base currency is supported")

    cash_floor = _fraction(raw.get("cash_floor", 0.02), "cash_floor", problems)

    raw_sleeves = raw.get("sleeves")
    sleeves: list[Sleeve] = []
    if not isinstance(raw_sleeves, list) or not raw_sleeves:
        problems.append("sleeves must be a non-empty list")
        raw_sleeves = []
    for i, item in enumerate(raw_sleeves):
        sleeve = _parse_sleeve(item, i, problems)
        if sleeve is not None:
            sleeves.append(sleeve)

    ids = [s.id for s in sleeves]
    for dup in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"sleeve id {dup!r} appears more than once")

    owners: dict[str, str] = {}
    for sleeve in sleeves:
        for inst in sleeve.all_instruments:
            if inst in owners and owners[inst] != sleeve.id:
                problems.append(f"instrument {inst} is in both {owners[inst]} and {sleeve.id}")
            owners[inst] = sleeve.id

    if sleeves:
        total = sum(s.target for s in sleeves)
        if abs(total - 1.0) > 1e-6:
            problems.append(f"sleeve targets sum to {total:.6f}, not 1")
        cash = [s for s in sleeves if s.is_cash]
        if len(cash) != 1:
            problems.append("exactly one sleeve must have id 'cash'")
        elif cash_floor is not None and cash[0].target + _TOL < cash_floor:
            problems.append(f"cash target {cash[0].target} is below the cash floor {cash_floor}")

    if problems:
        raise ModelError(source, problems)
    return Model(
        name=name,
        version=version,
        base_currency=base_currency,
        cash_floor=cash_floor,
        sleeves=tuple(sleeves),
        source=source,
    )


def _parse_sleeve(item: object, index: int, problems: list[str]) -> Sleeve | None:
    where = f"sleeves[{index}]"
    if not isinstance(item, dict):
        problems.append(f"{where} must be a mapping")
        return None
    unknown = set(item) - {"id", "target", "band", "instruments", "session_policy"}
    if unknown:
        problems.append(f"{where}: unknown keys {sorted(unknown)}")

    sleeve_id = item.get("id")
    if not isinstance(sleeve_id, str) or not sleeve_id:
        problems.append(f"{where}: id is required")
        return None
    where = f"sleeve {sleeve_id}"

    target = _fraction(item.get("target"), f"{where} target", problems)
    band = _parse_band(item.get("band"), where, problems)
    if band is None and sleeve_id != CASH_SLEEVE:
        problems.append(f"{where}: band is required")
    if band is not None and target is not None and band.width(target) >= target > 0:
        problems.append(
            f"{where}: band is as wide as the target, so the sleeve could never breach low"
        )

    instruments: dict[str, tuple[str, ...]] = {}
    raw_inst = item.get("instruments")
    if not isinstance(raw_inst, dict) or not raw_inst:
        problems.append(f"{where}: instruments must be a mapping of session to list")
    else:
        for key, names in raw_inst.items():
            if key not in INSTRUMENT_KEYS:
                problems.append(
                    f"{where}: unknown session {key!r} (use {', '.join(INSTRUMENT_KEYS)})"
                )
                continue
            if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
                problems.append(f"{where}: instruments.{key} must be a list of symbols")
                continue
            instruments[key] = tuple(names)
        if not any(instruments.values()):
            problems.append(f"{where}: no tradable instrument in any session")

    if sleeve_id == CASH_SLEEVE:
        other = {n for names in instruments.values() for n in names} - CASH_INSTRUMENTS
        if other:
            problems.append(
                f"{where}: cash sleeve can only hold {sorted(CASH_INSTRUMENTS)}, not {sorted(other)}"
            )

    policy = item.get("session_policy") or {}
    max_off = 0.25
    if not isinstance(policy, dict):
        problems.append(f"{where}: session_policy must be a mapping")
    else:
        if set(policy) - {"max_off_hours_pct"}:
            problems.append(
                f"{where}: unknown session_policy keys {sorted(set(policy) - {'max_off_hours_pct'})}"
            )
        value = _fraction(
            policy.get("max_off_hours_pct", 0.25), f"{where} max_off_hours_pct", problems
        )
        if value is not None:
            max_off = value

    if target is None:
        return None
    return Sleeve(sleeve_id, float(target), band, instruments, float(max_off))


def _parse_band(raw: object, where: str, problems: list[str]) -> Band | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw or set(raw) - {"abs", "rel"}:
        problems.append(f"{where}: band must be a mapping with abs and/or rel")
        return None
    abs_ = _fraction(raw.get("abs"), f"{where} band.abs", problems, optional=True)
    rel = _fraction(raw.get("rel"), f"{where} band.rel", problems, optional=True)
    if abs_ is None and rel is None:
        return None
    return Band(abs=abs_, rel=rel)


def _fraction(
    value: object, what: str, problems: list[str], optional: bool = False
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        problems.append(f"{what} must be a number between 0 and 1")
        return None
    return float(value)
