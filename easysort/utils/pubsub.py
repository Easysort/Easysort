import zenoh
import msgspec
from typing import Callable, Generic, Self, TypeVar

T = TypeVar('T')

ZENOH_SESSION: zenoh.Session | None = None


def zenoh_session() -> zenoh.Session:
    global ZENOH_SESSION
    if ZENOH_SESSION is None:
        ZENOH_SESSION = zenoh.open(zenoh.Config())
    return ZENOH_SESSION


class Publisher(Generic[T]):
    """Publisher class that publishes messages of a given type to a given topic via Zenoh.

    The type must be a type that can be serialized by msgspec, e.g. a dataclass.

    Example:
    ```
    @dataclass
    class MyData:
        x: int
        y: float
        z: str | None = None

    pub = Publisher("test/hello", MyData)
    pub.publish(MyData(x=1, y=2.0, z="hello"))
    ```
    """

    def __init__(self, topic: str, type_: type[T]) -> None:
        self._pub = zenoh_session().declare_publisher(topic)
        self._encoder = msgspec.msgpack.Encoder()
        self._type = type_

    def publish(self, data: T) -> None:
        if not isinstance(data, self._type):
            raise TypeError(f"Expected {self._type}, got {type(data)}")
        self._pub.put(self._encoder.encode(data))

    def close(self) -> None:
        self._pub.undeclare()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class Subscriber(Generic[T]):
    """Subscriber class that subscribes to a given topic and calls a callback function with the
    deserialized message.

    The type must match the type of the messages being published to the topic.

    Example:
    ```
    def callback(data: MyData):
        print(data)

    sub = Subscriber("test/hello", MyData, callback)
    time.sleep(10)  # Wait for messages to be received
    ```
    """

    def __init__(
        self, topic: str, type_: type[T], callback: Callable[[T], None]
    ) -> None:
        self._decoder = msgspec.msgpack.Decoder(type=type_)
        self._callback = callback
        self._sub = zenoh_session().declare_subscriber(topic, self._handle_message)

    def _handle_message(self, data: zenoh.Sample) -> None:
        try:
            decoded: T = self._decoder.decode(data.payload.to_bytes())
        except msgspec.DecodeError as e:
            raise ValueError(f"Failed to decode message on topic '{self._sub.key_expr}': {data}") from e
        self._callback(decoded)

    def close(self) -> None:
        self._sub.undeclare()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_) -> None:
        self.close()
