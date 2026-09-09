from abc import ABC, abstractmethod


class Eval(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def root_dir(self):
        return self.trainer.default_root_dir

    @abstractmethod
    def bind_data_context(self, *, normalizer):
        """Attach normalization and other data-owned evaluation state."""
        pass

    @abstractmethod
    def on_validation_start(self):
        pass

    @abstractmethod
    def on_validation_end(self):
        pass

    @abstractmethod
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
