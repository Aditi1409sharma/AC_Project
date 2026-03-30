from .delta import transform as delta
from .concat import transform as concat
from .bitslice import transform as bitslice
from .joint import transform as joint
from .raw import transform as raw
from .statistical import transform as statistical
from .word import transform as word

REPRESENTATIONS = {
    "delta": delta,
    "concat": concat,
    "bitslice": bitslice,
    "joint": joint,
    "statistical": statistical,
    "raw": raw,
    "word": word,
}