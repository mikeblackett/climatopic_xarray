import datetime as dt
from typing import TypeGuard

import pandas as pd
from pandas.core.generic import IntervalClosedType

type DatetimeLikeType = dt.datetime | pd.Timestamp | str


def is_datetime_index(index: pd.Index) -> TypeGuard[pd.DatetimeIndex]:
    """Check if the index is a datetime index."""
    return isinstance(index, pd.DatetimeIndex)


def is_date_offset(offset: object) -> TypeGuard[pd.DateOffset]:
    """Check if the object is a date offset object."""
    return isinstance(offset, pd.DateOffset)


def is_interval_closed_type(closed: str) -> TypeGuard[IntervalClosedType]:
    """Check if the closed string is a valid interval closed type."""
    return closed in {'both', 'neither', 'right', 'left'}
