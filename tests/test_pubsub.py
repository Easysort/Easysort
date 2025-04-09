from dataclasses import dataclass
import time

import pytest

from easysort.utils.pubsub import Publisher, Subscriber


@dataclass
class Data1:
    x: int
    y: float
    z: str | None = None


@dataclass
class Data2:
    a: Data1
    b: Data1


def test_pubsub():
    pub = Publisher("test/hello", Data2)
    received1: list[Data2] = []
    received2: list[Data2] = []
    expected: list[Data2] = []
    sub1 = Subscriber("test/hello", Data2, received1.append)
    sub2 = Subscriber("test/hello", Data2, received2.append)
    for i in range(5):
        data = Data2(
            a=Data1(x=i, y=float(i), z="hello"),
            b=Data1(x=i * 2, y=i / 2),
        )
        expected.append(data)
        pub.publish(data)

    with pytest.raises(TypeError):
        pub.publish(Data1(x=1, y=2.0, z="hello"))  # type: ignore

    t = time.time()
    while time.time() - t < 1:
        if len(received1) == 5 and len(received2) == 5:
            break
        time.sleep(0.05)

    assert received1 == expected
    assert received2 == expected

    sub1.close()
    sub2.close()
    pub.close()
