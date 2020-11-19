from mimic.utils.utils import at_most_n
import pytest
from typing import Union


@pytest.mark.parametrize("len_iterator,n", [(5, 3), (10, None)])
def test_at_most_n(len_iterator: int, n: Union[int, None]):
    iterator = [i for i in range(len_iterator)]
    idx = 0
    for idx, _ in enumerate(at_most_n(iterator, n)):
        pass
    if n:
        assert idx + 1 == n
    else:
        assert idx + 1 == len_iterator


if __name__ == '__main__':
    test_at_most_n(10, None)
