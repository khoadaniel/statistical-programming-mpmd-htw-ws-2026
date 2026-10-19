import pytest

from staycost import stay_cost


def test_three_nights_with_the_cleaning_fee():
    assert stay_cost(80, 3) == 265.0


def test_no_fee():
    assert stay_cost(80, 3, cleaning_fee=0) == 240.0


@pytest.mark.parametrize("price, nights", [(-1, 1), (80, 0)])
def test_impossible_stays_are_refused(price, nights):
    with pytest.raises(ValueError):
        stay_cost(price, nights)
