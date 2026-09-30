from abc import ABC, abstractmethod


class Eval(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def root_dir(self):
        return self.trainer.default_root_dir

    @abstractmethod
    def on_validation_start(self):
        pass

    @abstractmethod
    def on_validation_end(self):
        pass

    @abstractmethod
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        """``mode``: "metrics" / "video" / "both". An evaluator that renders
        nothing may ignore it, but it must accept it -- ModelWrapper passes it
        for every loader of a config that declares ``video_episodes``."""
        pass
