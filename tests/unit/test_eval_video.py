"""EvalVideo viz gating / fps and the prefixed evaluator wrapper."""

from types import SimpleNamespace

from egomimic.eval.eval_train_viz import PrefixedEvalVideo
from egomimic.eval.eval_video import EvalVideo


class _StubEvalVideo(EvalVideo):
    def compute_metrics_and_viz(self, batch, do_viz=True):
        raise NotImplementedError


def test_video_fps_compensates_for_world_size():
    # Distributed val strides an episode's frames by world_size on each rank;
    # playback fps must scale down to keep videos wall-clock real-time.
    ev = _StubEvalVideo.__new__(_StubEvalVideo)
    for world, expected in [(1, 30), (2, 15), (4, 8), (8, 4)]:
        ev.trainer = SimpleNamespace(world_size=world)
        assert ev._video_fps() == expected, (world, ev._video_fps())
    ev.trainer = SimpleNamespace()  # no world_size attr -> assume 1
    assert ev._video_fps() == 30


def test_viz_gate_follows_lightning_epoch_convention():
    ev = _StubEvalVideo(viz_every_n_epochs=20)
    # Lightning validates at current_epoch 19, 39, ... for check_val_every_n_epoch=20
    for epoch, expected in [(0, False), (19, True), (20, False), (39, True)]:
        ev.trainer = SimpleNamespace(current_epoch=epoch)
        assert ev._should_viz() is expected, epoch
    ev.viz_every_n_epochs = 0
    assert ev._should_viz() is False


def test_viz_gate_always_renders_the_final_epoch():
    ev = _StubEvalVideo(viz_every_n_epochs=100)
    # trainer=debug: 4 epochs, val at current_epoch 1 and 3 -> video only at 3
    for epoch, expected in [(1, False), (3, True)]:
        ev.trainer = SimpleNamespace(current_epoch=epoch, max_epochs=4)
        assert ev._should_viz() is expected, epoch
    # eval mode validates once at epoch 0 with max_epochs=1
    ev.trainer = SimpleNamespace(current_epoch=0, max_epochs=1)
    assert ev._should_viz() is True
    # unbounded runs (max_epochs None / -1) only follow the interval
    for max_epochs in (None, -1):
        ev.trainer = SimpleNamespace(current_epoch=0, max_epochs=max_epochs)
        assert ev._should_viz() is False
    # viz_every_n_epochs=0 still turns video off entirely
    ev.viz_every_n_epochs = 0
    ev.trainer = SimpleNamespace(current_epoch=3, max_epochs=4)
    assert ev._should_viz() is False


def test_train_viz_wrapper_prefixes_and_forwards():
    class _Base(EvalVideo):
        def __init__(self):
            super().__init__(viz_func={}, transform_lists={}, viz_every_n_epochs=7)
            self.seen = None

        def compute_metrics_and_viz(self, batch, do_viz=True):
            self.seen = (self.model, do_viz)
            return {"Valid/x": 1.0}, {}

    base = _Base()
    tv = PrefixedEvalVideo(base)
    algo = object()
    tv.model = algo  # property setter forwards to base too
    metrics, _ = tv.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"train_viz/Valid/x"}, metrics
    assert base.seen == (algo, False)
    assert tv.viz_every_n_epochs == 7, "wrapper inherits the base viz gate"
    tv.trainer = SimpleNamespace(default_root_dir="/tmp/run")
    assert base.trainer is tv.trainer
    assert tv.video_dir().endswith("videos_train_viz")

    op = PrefixedEvalVideo(base, prefix="unseen_op_valid")
    metrics, _ = op.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"unseen_op_valid/Valid/x"}, metrics
    op.trainer = SimpleNamespace(default_root_dir="/tmp/run")
    assert op.video_dir().endswith("videos_unseen_op_valid")
