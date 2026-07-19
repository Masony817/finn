"""Plugin resolution: built-ins first, then dotted paths.

Extend scopik without touching it: implement the relevant Protocol in your own
package and reference it from your profile as "mypkg.sources:MyLogSource".
(A registry/entry-point mechanism is deliberately deferred until there is a
second user to need it.)
"""

from __future__ import annotations

import importlib
from typing import Any

from scopik.datamodel import ScopikError


def builtin_sources() -> dict[str, Any]:
    from scopik.sources.csv_source import CsvSource
    from scopik.sources.prefixed_csv import PrefixedCsvSource

    return {"csv": CsvSource, "prefixed_csv": PrefixedCsvSource}


def resolve(kind: str, name: str, builtins: dict[str, Any]) -> Any:
    """Return an instance of the named plugin class."""

    if name in builtins:
        return builtins[name]()

    if ":" in name:
        module_name, _, attr = name.partition(":")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)()
        except (ImportError, AttributeError) as exc:
            raise ScopikError(f"cannot load {kind} plugin {name!r}: {exc}") from exc

    known = ", ".join(sorted(builtins)) or "<none>"
    raise ScopikError(
        f"unknown {kind} {name!r} (built-ins: {known}; or use a dotted path 'pkg.mod:Class')"
    )


def resolve_source(name: str) -> Any:
    return resolve("source", name, builtin_sources())
