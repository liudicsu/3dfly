"""Binary brain/body frame stream (protocol v1) for custom front ends.

* ``protocol``  — versioned binary frame format (encode / decode),
* ``builder``   — turns closed-loop sim state into frames (window-max
  decimation so a GF burst between display frames is not lost),
* ``server``    — websocket server (own thread + asyncio loop) that also
  serves a small browser inspector,
* ``client``    — tiny Python client / decoder.

See ``docs/stream_protocol.md`` for the wire format.
"""

from threedfly.stream.protocol import (
    MAGIC,
    PROTOCOL_VERSION,
    Frame,
    MsgType,
    Section,
    decode_message,
    encode_frame,
    encode_hello,
)

__all__ = [
    "MAGIC",
    "PROTOCOL_VERSION",
    "Frame",
    "MsgType",
    "Section",
    "decode_message",
    "encode_frame",
    "encode_hello",
]
