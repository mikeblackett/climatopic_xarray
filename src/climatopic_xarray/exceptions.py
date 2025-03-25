class ClimatopicError(Exception):
    """Base class for errors in the climatopic_xarray package."""

    pass


class InvalidArgumentError(ClimatopicError):
    """Error raised when an invalid argument is encountered."""

    pass


class NotYetImplementedError(ClimatopicError):
    """
    Error raised when a feature is not yet implemented.

    This is not the same as `NotImplementedError`, which is used when a method
    is not implemented in a base class.
    """

    pass
