import os

from tensorboard.compat import tf2 as tf


class SummaryWriter:
    def __init__(self, log: str):
        if log is None:
            self.writer = None
        else:
            self.writer = {
                "gen": tf.summary.create_file_writer(log),
                "train": tf.summary.create_file_writer(os.path.join(log, "train")),
                "val": tf.summary.create_file_writer(os.path.join(log, "val")),
            }

    def add_history(self, history: dict[str, list[float]], step: int):
        if self.writer is not None:
            with self.writer["train"].as_default():
                tf.summary.scalar("loss", history["train"][-1], step=step)
            with self.writer["val"].as_default():
                tf.summary.scalar("loss", history["val"][-1], step=step)
            with self.writer["gen"].as_default():
                tf.summary.scalar("learning_rate", history["lr"][-1], step=step)

    def scalar(self, name: str, value: float, step: int, tag: str = "gen"):
        if self.writer is not None:
            with self.writer[tag].as_default():
                tf.summary.scalar(name, value, step=step)

    def text(self, name, data, step=None, tag: str = "gen"):
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
        tag: str = "gen",
    ):
        if self.writer is not None:
            with self.writer[tag].as_default():
                tf.summary.histogram(
                    name, data, step=step, buckets=bins, description=description
                )
