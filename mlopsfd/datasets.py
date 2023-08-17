import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from typing import Callable, Iterable, List, Protocol


GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
SEED = 1234


@dataclass
class ArchimedianRadius:
    factor: float = 1.0
    
    def __call__(self, phi):
        return self.factor * phi


@dataclass
class Spiral:
    
    radius: Callable[[float], float]
    shift: float = 0.0

    def get_x_y(self, angle: float):
        r = self.radius(angle)
        x = r * np.cos(angle + self.shift)
        y = r * np.sin(angle + self.shift)
        
        return x, y


class Distribution(Protocol):
    
    def generate(self, rng: np.random.Generator, size: tuple = None) -> np.ndarray: ...


@dataclass
class Normal(Distribution):
    loc: float = 0.0
    scale: float = 1.0    

    def generate(self, rng: np.random.Generator, size: tuple = None) -> np.ndarray:
        return rng.normal(self.loc, self.scale, size=size)


fr = ArchimedianRadius(GOLDEN_RATIO)

SPIRAL0 = Spiral(fr)
SPIRAL1 = Spiral(fr, shift=2/3*np.pi)
SPIRAL2 = Spiral(fr, shift=4/3*np.pi)
SPIRALS = [SPIRAL0, SPIRAL1, SPIRAL2]


def generate_spirals(
    spirals: Iterable[Spiral] = None,
    angle: np.ndarray = None,
    rng: np.random.Generator = None,
    distribution: Distribution = None,
) -> pd.DataFrame:
    
    if spirals is None:
        spirals = SPIRALS

    if angle is None:
        angle = np.linspace(0.0, 5.0, num=500)
    
    if rng is None:
        rng = np.random.default_rng(SEED)

    if distribution is None:
        distribution = Normal(0.0, 1.0)

    xys = [spiral.get_x_y(angle) for spiral in spirals]
    eps = distribution.generate(rng, size=(len(spirals), 2, len(angle)))

    dfs = [
        pd.DataFrame({'X1': x + eps[i, 0], 'X2': y + eps[i, 1], 'color': f'c{i+1}'})
        for (i, (x, y)) in enumerate(xys)
    ]
    df = pd.concat(dfs, axis=0)

    return df
