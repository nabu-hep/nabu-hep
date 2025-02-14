import os
import time

from tensorboardX import SummaryWriter as SummaryWriterX


class SummaryWriter:
    """Summary writer for tensorboard"""

    def __init__(self, log: str, tag: str = "summary"):
        if log is None:
            self.writer = None
        else:
            self.writer = SummaryWriterX(os.path.join(log, tag))

    def scalar(self, name: str, value: float, step: int = 0) -> None:
        """
        Add scalar to tensorboard

        Args:
            name (``str``): name of the scalar
            value (``float``): value of the scalar
            step (``int``): step
            tag (``str``, default ``"summary"``):  tag
        """
        if self.writer is not None:
            self.writer.add_scalar(name, value, global_step=step, walltime=time.time())

    def text(self, name: str, data: str, step: int = 0) -> None:
        """
        Add text to tensorboard

        Args:
            name (``str``): name of the text
            data (``str``): data
            step (``int``, default ``None``): step
            tag (``str``, default ``"summary"``): tag
        """
        if self.writer is not None:
            self.writer.add_text(name, data, global_step=step, walltime=time.time())

    def figure(self, name: str, figure, step: int = 0) -> None:
        """
        Add figure to tensorboard

        Args:
            name (``str``): name of the figure
            figure (``plt.Figure``): figure
            step (``int``, default ``None``): step
            tag (``str``, default ``"summary"``): tag
        """
        if self.writer is not None:
            self.writer.add_figure(name, figure, global_step=step, walltime=time.time())

    def close(self) -> None:
        """Close the writer"""
        if self.writer is not None:
            self.writer.close()
