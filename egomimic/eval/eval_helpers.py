"""Shared helpers for evaluation implementations."""


def visualize_predictions(viz_func, predictions, batch):
    """Render predictions with the configured per-embodiment visualizer."""
    if viz_func is None:
        raise ValueError("viz_func is not set")
    from egomimic.rldb.embodiment.embodiment import get_embodiment

    embodiment_id = batch["embodiment"][0].item()
    embodiment_name = get_embodiment(embodiment_id).lower()
    return viz_func[embodiment_name](predictions, batch)
