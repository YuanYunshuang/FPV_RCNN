from collections import defaultdict, deque
import torch
from datetime import datetime
from functools import partial


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class LogMeter(object):
    def __init__(self, total_iter, log_path, delimiter="\t", log_every=20, wandb_project=None):
        self.meters = defaultdict(partial(SmoothedValue, fmt="{avg:.4f}"))
        file_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".txt"
        self.log_fh = (log_path / file_name).open('a')
        self.delimiter = delimiter
        self.log_every = log_every
        self.log_msg = self.delimiter.join([
            'E:{epoch:2d}',
            'I:[{itr:4d}/' + str(total_iter) + ']',
            'lr:{lr:.6f}',
            '{meters}'
        ])
        if wandb_project is not None:
            import wandb
            wandb.init(project=wandb_project)
            wandb.config.log_histo = True
            wandb.config.step = 0
            wandb_project = wandb
        self.wandb = wandb_project

    def update(self, kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_fh.close()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log(self, epoch, iteration, lr, **kwargs):
        self.update(kwargs)
        if iteration % self.log_every==0:
            msg = self.log_msg.format(
                epoch=epoch,
                itr=iteration,
                lr=lr,
                meters=str(self)
            )
            print(msg)
            self.log_fh.write(msg + "\n")
            if self.wandb is not None:
                self.wandb.log({('avg/' + name): meter.avg for name, meter in self.meters.items()})
                self.wandb.log({('global_avg/' + name): meter.global_avg for name, meter in self.meters.items()})

