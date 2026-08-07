from abc import ABC, abstractmethod


class Eval(ABC):
    """Base evaluator.

    Subclasses implement compute_metrics_and_viz. The lifecycle hooks
    (on_validation_start/step/end) have no-op defaults so subclasses
    only override what they need.

    The lightning trainer + algo model are injected by the caller
    (e.g. ModelWrapper.on_validation_*) before any lifecycle hook
    fires; we hold them as plain attributes so any subclass can read
    them via self.trainer / self.model.
    """

    def __init__(self):
        self.trainer = None
        self.model = None
        # Trainer-config overrides consumed by trainHydra in `mode=eval`.
        # Video evaluators (EvalVideo) repopulate this; the base default of
        # {} lets non-video evaluators (e.g. the EvalList composite runner)
        # run in eval mode without crashing on a missing attribute.
        self.override_dict = {}

    def root_dir(self):
        return self.trainer.default_root_dir

    @abstractmethod
    def compute_metrics_and_viz(self, batch):
        """Return (metrics_dict, images_dict)."""
        raise NotImplementedError

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass
