"""Registry of available :class:`~vwf.sources.base.ObservationSource` adapters.

Adapters are keyed by their ``name`` so that a country can carry more than one
source (for example a turbine-level and a country-level adapter for the same
region). :func:`resolve` picks the right one for a ``(country, obs_level)`` pair.
"""
from __future__ import annotations

from typing import Any

from vwf.sources.base import ObsLevel, ObservationSource

_REGISTRY: dict[str, type[ObservationSource]] = {}


def register(cls: type[ObservationSource]) -> type[ObservationSource]:
    """Class decorator adding an adapter to the registry.

    Args:
        cls: The :class:`~vwf.sources.base.ObservationSource` subclass to
            register, keyed by its ``name`` class attribute.

    Returns:
        ``cls`` itself, unchanged, so the decorator form
        ``@register`` above a class definition works.

    Raises:
        TypeError: If ``cls`` is not an ObservationSource subclass.
        ValueError: If ``name`` or ``obs_level`` is missing, or ``name`` is taken.
    """
    if not (isinstance(cls, type) and issubclass(cls, ObservationSource)):
        raise TypeError(f"{cls!r} is not an ObservationSource subclass")

    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")

    obs_level = getattr(cls, "obs_level", None)
    if obs_level not in ("turbine", "country"):
        raise ValueError(
            f"{cls.__name__}.obs_level must be 'turbine' or 'country', got {obs_level!r}"
        )

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"An observation source named {name!r} is already registered")

    _REGISTRY[name] = cls
    return cls


def available_sources() -> tuple[str, ...]:
    """Return the names of every registered adapter, sorted."""
    return tuple(sorted(_REGISTRY))


def get_source(name: str, *args: Any, **kwargs: Any) -> ObservationSource:
    """Construct a registered adapter by name.

    Arguments are forwarded verbatim to the adapter's constructor, whose
    signature varies by adapter (a country code for auto-resolvable sources, the
    data frames themselves for :class:`InMemoryCountrySource`), which is why they
    are untyped here.

    Args:
        name: Registry key of the adapter, e.g. ``"european-turbine"`` (see
            :func:`available_sources`).
        *args: Positional arguments for the adapter's constructor.
        **kwargs: Keyword arguments for the adapter's constructor.

    Returns:
        The constructed adapter instance.

    Raises:
        KeyError: If no adapter is registered under ``name``.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown observation source {name!r}. Registered: {available_sources()}"
        ) from None
    return cls(*args, **kwargs)


def resolve(country: str, obs_level: ObsLevel) -> ObservationSource:
    """Find the adapter that serves ``country`` at ``obs_level``.

    Args:
        country: Country code, case-insensitive (e.g. ``"dk"``).
        obs_level: ``"turbine"`` or ``"country"``; an adapter is only
            considered if its own ``obs_level`` matches.

    Returns:
        A ready-to-use adapter instance, constructed with the country code.

    Raises:
        NotImplementedError: If ``obs_level`` is ``"country"`` and no adapter
            claims the country. Country-level observations are supplied by the
            caller, so there is nothing to fall back on.
        ValueError: If ``obs_level`` is ``"turbine"`` and no adapter claims the
            country.
    """
    country = country.upper()

    for cls in _REGISTRY.values():
        if cls.obs_level == obs_level and country in cls.countries:
            return cls(country)

    if obs_level == "country":
        raise NotImplementedError(
            f"No country-level observation source is registered for {country!r}. "
            "Country-level observations must be supplied by the caller, either via "
            "PyVWF.load_country_data(grid_points, obs_train, obs_test) or by passing "
            "source=InMemoryCountrySource(...) to train_set/val_set. Alternatively, "
            "register a custom ObservationSource with obs_level='country'."
        )

    raise ValueError(
        f"Unsupported country={country} at obs_level='turbine'. "
        f"Registered sources: {available_sources()}"
    )
