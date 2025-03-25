import pytest

from climatopic_xarray.utilities import mapping_or_kwargs


@pytest.mark.parametrize(
    'pargs, kwargs, expected',
    [
        (None, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}),
        ({'a': 1, 'b': 2}, {}, {'a': 1, 'b': 2}),
        ({}, {'a': 1, 'b': 2}, {'a': 1, 'b': 2}),
    ],
)
def test_mapping_or_kwargs(pargs, kwargs, expected):
    actual = mapping_or_kwargs(
        parg=pargs, kwargs=kwargs, func_name='test_func'
    )
    assert actual == expected


@pytest.mark.parametrize(
    'pargs, kwargs, func_name',
    [
        ({'a': 1}, {'b': 2}, 'test_func'),
    ],
)
def test_mapping_or_kwargs_raises_value_error(pargs, kwargs, func_name):
    with pytest.raises(ValueError):
        mapping_or_kwargs(pargs, kwargs, func_name)
