import os

from tensorboard.compat import tf2 as tf


class SummaryWriter:
    """Summary writer for tensorboard"""

    def __init__(self, log: str):
        if log is None:
            self.writer = None
        else:
            self.writer = {
                "summary": tf.summary.create_file_writer(log),
                "train": tf.summary.create_file_writer(os.path.join(log, "train")),
                "val": tf.summary.create_file_writer(os.path.join(log, "val")),
            }

    def add_history(self, history: dict[str, list[float]], step: int) -> None:
        """
        Add history to tensorboard

        Args:
            history (``dict[str, list[float]]``): history of the training
            step (``int``): current step
        """
        if self.writer is not None:
            with self.writer["train"].as_default():
                tf.summary.scalar("loss", history["train"][-1], step=step)
            with self.writer["val"].as_default():
                tf.summary.scalar("loss", history["val"][-1], step=step)
            with self.writer["summary"].as_default():
                tf.summary.scalar("learning_rate", history["lr"][-1], step=step)

    def scalar(self, name: str, value: float, step: int, tag: str = "summary") -> None:
        """
        Add scalar to tensorboard

        Args:
            name (``str``): name of the scalar
            value (``float``): value of the scalar
            step (``int``): step
            tag (``str``, default ``"summary"``):  tag
        """
        if self.writer is not None:
            with self.writer[tag].as_default():
                tf.summary.scalar(name, value, step=step)

    def text(self, name: str, data: str, step: int = None, tag: str = "summary") -> None:
        """
        Add text to tensorboard

        Args:
            name (``str``): name of the text
            data (``str``): data
            step (``int``, default ``None``): step
            tag (``str``, default ``"summary"``): tag
        """
        if self.writer is not None:
            with self.writer[tag].as_default():
                tf.summary.text(name, data, step=step)

    def histogram(
        self,
        name: str,
        data: list[float],
        step: int,
        bins: int = 30,
        description: str = None,
        tag: str = "summary",
    ) -> None:
        """
        Add histogram to tensorboard

        Args:
            name (``str``): name of the histogram
            data (``list[float]``): data
            step (``int``): step
            bins (``int``, default ``30``): number of bins
            description (``str``, default ``None``): description
            tag (``str``, default ``"summary"``): tag
        """
        if self.writer is not None:
            with self.writer[tag].as_default():
                tf.summary.histogram(
                    name, data, step=step, buckets=bins, description=description
                )
