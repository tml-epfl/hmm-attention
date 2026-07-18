from difflib import get_close_matches
from typing import Any, Dict, ItemsView, Iterator, KeysView, ValuesView


class MetricRegistry:
    """Dict-like container for the trainer's metrics with strict access.

    Two safety wins over a plain dict:

    - `register(key, metric)` refuses duplicates. Catches double-registration
      caused by copy-pasted metric setup blocks.
    - `metrics[key]` raises `KeyError` with a "did-you-mean" suggestion when
      the key isn't registered. Catches typos that would otherwise silently
      write to a new key (and never get read).

    Iteration / `.items()` / `.values()` mirror dict for drop-in use in
    W&B logging and end-of-step reset loops.
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def register(self, key: str, metric: Any) -> None:
        if key in self._store:
            raise ValueError(f"Metric {key!r} is already registered")
        self._store[key] = metric

    def __getitem__(self, key: str) -> Any:
        try:
            return self._store[key]
        except KeyError:
            close = get_close_matches(key, self._store.keys(), n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(map(repr, close))}?" if close else ""
            raise KeyError(f"Metric {key!r} is not registered.{hint}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> KeysView[str]:
        return self._store.keys()

    def values(self) -> ValuesView[Any]:
        return self._store.values()

    def items(self) -> ItemsView[str, Any]:
        return self._store.items()

    def pop(self, key: str, default: Any = None) -> Any:
        return self._store.pop(key, default)
