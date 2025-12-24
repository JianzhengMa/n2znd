from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


class HistoryWindow(List[T]):
    """Fixed-size rolling buffer used for trajectory histories."""

    def __init__(self, keep: int):
        super().__init__()
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self.keep = keep

    def append(self, value: T) -> None:  # type: ignore[override]
        super().append(value)
        while len(self) > self.keep:
            del self[0]

    def seed(self, values: Sequence[T]) -> None:
        """Reset the buffer using the provided values."""
        self.clear()
        for value in values[-self.keep :]:
            super().append(value)

    def extend_from(self, iterable: Iterable[T]) -> None:
        for value in iterable:
            self.append(value)

    def window(self) -> Sequence[T]:
        if len(self) < self.keep:
            raise ValueError("history does not yet contain enough elements")
        return tuple(self)
