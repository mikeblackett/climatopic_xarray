import calendar
from collections.abc import Iterable, Sequence
from typing import Literal

from hypothesis.errors import InvalidArgument


type OffsetCategoryName = Literal[
    'D',  # day
    'M',  # month
    'MS',  # month-start
    'ME',  # month-end
    'Q',  # quarter
    'QS',  # quarter-start
    'QE',  # quarter-end
    'Y',  # year
    'YS',  # year-start
    'YE',  # year-end
    'W',  # week
]
type OffsetCategories = Sequence[OffsetCategoryName]
type CategoriesTuple = tuple[OffsetCategoryName, ...]
type OffsetStrings = tuple[str, ...]

SUPER_CATEGORIES = ('M', 'Q', 'Y')
CATEGORIES = (
    'D',
    'M',
    'MS',
    'ME',
    'Q',
    'QS',
    'QE',
    'Y',
    'YS',
    'YE',
    'W',
)


def _add_month_abbr(prefix: Iterable[str]) -> tuple[str, ...]:
    return tuple(f'{prefix}-{m.upper()}' for m in calendar.month_abbr if m)


def _add_day_abbr(prefix: Iterable[str]) -> tuple[str, ...]:
    return tuple(f'{prefix}-{d.upper()}' for d in calendar.day_abbr if d)


FREQ_MAP: dict[OffsetCategoryName, OffsetStrings] = {
    'D': ('D',),
    'M': ('MS', 'ME'),
    'MS': ('MS',),
    'ME': ('ME',),
    'Q': ('QS', 'QE'),
    'QS': ('QS', *_add_month_abbr('QS')),
    'QE': ('QE', *_add_month_abbr('QE')),
    'Y': ('YS', 'YE'),
    'YS': ('YS', *_add_month_abbr('YS')),
    'YE': ('YE', *_add_month_abbr('YE')),
    'W': ('W', *_add_day_abbr('W')),
}


def get_categories(categories: OffsetCategories) -> CategoriesTuple:
    out = set(categories)
    for c in categories:
        if c in SUPER_CATEGORIES:
            # Super categories are expanded to include all their subcategories
            out.update(x for x in CATEGORIES if x.startswith(c))
            out.discard(c)
        elif c not in CATEGORIES:
            raise InvalidArgument(f'Unknown category: {c}')
    # Return in the order of CATEGORIES
    return tuple(c for c in CATEGORIES if c in out)


def get_freqs(categories: OffsetCategories) -> OffsetStrings:
    cats = get_categories(categories)
    out = []
    for c in cats:
        out.extend(FREQ_MAP[c])
    return tuple(out)


def query_offset_strings(
    categories: OffsetCategories | None = None,
    exclude_categories: OffsetCategories | None = None,
    exclude_freqs: Iterable[str] | None = None,
    include_freqs: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if categories and exclude_categories:
        raise InvalidArgument(
            f'Pass one of {categories=!r} and {exclude_categories=!r}.'
        )
    categories = get_categories(categories or CATEGORIES)
    exclude_categories = (
        get_categories(exclude_categories) if exclude_categories else None
    )
    if exclude_categories:
        categories = [c for c in categories if c not in exclude_categories]
    out = set(get_freqs(categories))
    include_freqs = set(include_freqs) if include_freqs else set()
    exclude_freqs = set(exclude_freqs) if exclude_freqs else set()
    if overlap := set(include_freqs) & set(exclude_freqs):
        raise InvalidArgument(
            f'Freqs {overlap!r} are present in both include_freqs={include_freqs!r}'
            f' and exclude_characters={exclude_freqs!r}.'
        )
    if include_freqs:
        out |= include_freqs
    if exclude_freqs:
        out -= exclude_freqs
    return tuple(out)
