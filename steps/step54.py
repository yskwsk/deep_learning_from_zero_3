import numpy as np

from dezero import test_mode
import dezero.functions as F


def test_dropout1() -> None:
    dropout_ratio = 0.6
    x = np.ones(10)

    mask = np.random.rand(10) > dropout_ratio
    y = x * mask
    print(y)


def main() -> None:
    x = np.ones(5)
    print(x)

    # 学習時
    y = F.dropout(x)
    print(y)

    # テスト時
    with test_mode():
        y = F.dropout(x)
        print(y)


if __name__ == "__main__":
    test_dropout1()

    main()
