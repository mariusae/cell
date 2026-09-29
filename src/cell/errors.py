"""Errors raised by the cell runtime itself (as opposed to by cell bodies)."""


class CellError(Exception):
    """Base class for runtime errors."""


class DataError(CellError, TypeError):
    """A value that must be data (DESIGN §1.2) is not."""


class ContextError(CellError):
    """A ctx was used incorrectly: missing, finished, or from the wrong task."""


class DeterminismError(CellError):
    """Replay found a different effectful call at a journaled position (DESIGN §3)."""
