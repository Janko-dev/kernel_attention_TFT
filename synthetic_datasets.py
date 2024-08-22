import torch
from torch.utils.data import Dataset


class BaseTimeSeriesDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        fX: torch.Tensor,
        seq_len: int,
        shift: int
    ):
        """
        :param X: time steps (covariates)
        :param fX: features per time step
        :param seq_len: length of each sequence example in dataset
        :param shift: number of steps to shift the target values
        """
        self.seq_len = seq_len
        self.shift = shift
        self.X = X
        self.fX = fX
        self.mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)

    def __getitem__(self, index):

        sample = (
            self.X[index : index + self.seq_len],
            self.fX[index : index + self.seq_len],
            self.X[index + self.seq_len : index + self.seq_len + self.shift],
            self.fX[index + self.seq_len : index + self.seq_len + self.shift],
        )
        return sample

    def __len__(self):
        return len(self.X) - self.seq_len - self.shift

class StepSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha: float = 0.1,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = torch.sign(torch.sin(alpha * X))
        super().__init__(X, fX, seq_len, shift)


class DecayingStepSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha1: float = 0.1,
        alpha2: float = 0.001,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = torch.sign(torch.sin(alpha1 * X)) * torch.exp(-alpha2 * X)
        super().__init__(X, fX, seq_len, shift)


class MultipleStepSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha1: float = 0.1,
        alpha2: float = 0.05,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = (
            torch.sign(torch.sin(alpha1 * X)) + torch.sign(torch.sin(alpha2 * X)) + 3
        )  # ensure not close to zero
        super().__init__(X, fX, seq_len, shift)


class SineSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha: float = 0.1,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = torch.sin(alpha * X)
        super().__init__(X, fX, seq_len, shift)


class DecayingSineSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha1: float = 0.1,
        alpha2: float = 0.001,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = torch.sin(alpha1 * X) * torch.exp(-alpha2 * X)
        super().__init__(X, fX, seq_len, shift)


class MultipleSineSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        alpha1: float = 0.1,
        alpha2: float = 0.05,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        fX = torch.sin(alpha1 * X) + torch.sin(alpha2 * X)
        super().__init__(X, fX, seq_len, shift)


class SawtoothSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        p: float = 5,
        alpha: float = 0.1,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        t = alpha * X
        fX = 2 * (t / p - torch.floor(0.5 + t / p)) + 3  # ensure not close to zero
        super().__init__(X, fX, seq_len, shift)


class DecayingSawtoothSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        p: float = 5,
        alpha1: float = 0.1,
        alpha2: float = 0.001,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        t = alpha1 * X
        fX = (
            2 * (t / p - torch.floor(0.5 + t / p)) * torch.exp(-alpha2 * X) + 3
        )  # ensure not close to zero
        super().__init__(X, fX, seq_len, shift)


class MultipleSawtoothSyntheticDataset(BaseTimeSeriesDataset):
    def __init__(
        self,
        len_timeseries: int,
        seq_len: int,
        shift: int,
        p: float = 5,
        alpha1: float = 0.1,
        alpha2: float = 0.05,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)
        t1 = alpha1 * X
        t2 = alpha2 * X
        fX1 = 2 * (t1 / p - torch.floor(0.5 + t1 / p))
        fX2 = 2 * (t2 / p - torch.floor(0.5 + t2 / p))
        fX = fX1 + fX2 + 3  # ensure not close to zero
        super().__init__(X, fX, seq_len, shift)


class PeriodicSyntheticDataset(BaseTimeSeriesDataset):
    """
    :param len_timeseries: total length of entire time series
    :param A1, A2, A3: Control the timeseries periodic curves
    """
    def __init__(
        self,
        len_timeseries,
        seq_len,
        shift,
        A1,
        A2,
        A3,
        A4,
    ):

        X = torch.arange(0, len_timeseries, dtype=torch.float32).reshape(-1, 1)

        c1 = A1 * torch.sin(torch.pi * X[:12] / 6) + 72
        c2 = A2 * torch.sin(torch.pi * X[12:24] / 6) + 72
        c3 = A3 * torch.sin(torch.pi * X[24:96] / 6) + 72
        c4 = A4 * torch.sin(torch.pi * X[96:120] / 12) + 72

        fX = torch.cat([c1, c2, c3, c4], dim=0)  # concat components
        fX = fX.repeat((len_timeseries + 120) // 120, 1)  # periodic repeats
        fX = fX[:len_timeseries]  # constrain total length
        fX = fX + torch.randn(fX.shape)  # add noise

        super().__init__(X, fX, seq_len, shift)


"""
simple Synthetic dataset with a single timeseries
"""
class SimplePeriodicDataset(PeriodicSyntheticDataset):
    def __init__(
        self,
        len_timeseries=120,
        seq_len=96,
        shift=1,
        A1=40,
        A2=60,
        A3=3,
    ):
        A1, A2, A3 = torch.tensor([[A1], [A2], [A3]])
        A4 = torch.max(A1, A2)
        super().__init__(len_timeseries, seq_len, shift, A1, A2, A3, A4)


"""
Complex Synthetic dataset with more uncertainty and many timeseries.
Produces a multi-variate time series dataset that repeats itself every 120 steps.
"""
class MultivarPeriodicDataset(PeriodicSyntheticDataset):
    def __init__(
        self,
        len_timeseries=120,
        seq_len=96,
        shift=24,
        n_timeseries=100,
    ):
        A1, A2, A3 = torch.rand(3, 1, n_timeseries) * 60
        A4 = torch.max(A1, A2)
        super().__init__(len_timeseries, seq_len, shift, A1, A2, A3, A4)
