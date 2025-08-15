from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import numpy as np


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return b64encode(buf.read()).decode()


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
