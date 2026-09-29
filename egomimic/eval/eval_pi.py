from egomimic.eval.eval_video import EvalVideo


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models: EvalVideo's metrics as is. Per embodiment, on
    the model's native (unnormalized) output:
      - ``action_loss``: flow-matching val loss, same as training (also logged
        per embodiment as ``<name>_loss``)
      - the layout metrics of :mod:`egomimic.eval.action_metrics`:
        cartesian widths get ``xyz_paired_mse_avg`` (m^2),
        ``rot_err_deg_avg`` / ``rot_err_deg_final`` (geodesic, after
        Gram-Schmidt), ``xyz_l2_{early,mid,late}`` (m), ``xyz_dtw_avg`` and
        ``xyz_frechet_gauss_avg``; keypoint widths get ``kp_l2_avg`` /
        ``kp_l2_{early,mid,late}`` (m), ``wrist_xyz_paired_mse_avg``,
        ``wrist_xyz_l2_*``, ``wrist_rot_err_deg_*`` and ``kp_dtw_avg``.

    Metrics are frame-invariant (the cam-frame revert is a rigid transform
    shared by prediction and gt), so the ``transform_lists`` revert runs only
    when rendering the viz video (``do_viz``), never for metrics.
    """
