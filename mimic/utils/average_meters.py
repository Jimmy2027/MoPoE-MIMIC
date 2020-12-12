import typing

import numpy as np


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Taken from https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def get_average(self):
        return self.val


class AverageMeterNestedDict:
    """
    Computes and stores the average and current value
    Inspired by https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str, structure: typing.Mapping[str, typing.Mapping[str, typing.Iterable[None]]]):
        self.name = name
        self.structure = structure
        self.vals: typing.Mapping[str, typing.Mapping[str, typing.Iterable[typing.Optional[float]]]] = structure

    def update(self, val: typing.Mapping[str, typing.Mapping[str, typing.Iterable[float]]]) -> None:
        for k1 in self.structure:
            for k2 in self.structure:
                self.vals[k1][k2].append(val[k1][k2])

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, float]]:
        d = {}
        for k1 in self.structure:
            for k2 in self.structure:
                d[k1][k2] = np.mean(self.vals[k1][k2])

        return d


class AverageMeterDict:
    """
    Computes and stores the average and current value
    Inspired by https://github.com/pytorch/examples/blob/a3f28a26851867b314f4471ec6ca1c2c048217f1/imagenet/main.py#L363
    """

    def __init__(self, name: str):
        self.name = name
        self.vals: typing.Optional[typing.Mapping[str, typing.Iterable[float]]] = None

    def update(self, val: typing.Mapping[str, typing.Iterable[float]]) -> None:
        if not self.vals:
            self.vals = {k: [] for k in val}
        for key in val:
            self.vals[key].append(val[key])

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, float]]:
        return {key: np.mean(self.vals[key]) for key in self.vals}


class AverageMeterLatents(AverageMeterDict):
    def __init__(self, name: str):
        super().__init__(name=name)

    def update(self, val: typing.Mapping[str, typing.Tuple[typing.Iterable[float], typing.Iterable[float]]]):
        if not self.vals:
            self.vals = {k: ([], []) for k in val}
        for key in val:
            self.vals[key][0].append(val[key][0].mean().item())
            self.vals[key][1].append(val[key][1].mean().item())

    def get_average(self) -> typing.Mapping[str, typing.Mapping[str, typing.Tuple[float, float]]]:
        return {key: (np.mean(self.vals[key][0]), np.mean(self.vals[key][1])) for key in self.vals}