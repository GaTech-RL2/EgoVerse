import re, sys
base = open("egomimic/hydra_configs/model/bf_prdec.yaml").read()
SNAP = "/coc/flash7/paphiwetsa3/projects/EgoVerse-batchflow/eval_bf/bf_prdec_ctrl_ep{E}_snap.ckpt"

def make(name, epoch, ratio_w, ste_gain):
    s = base
    # weights-only init from the pinned ctrl snap
    s = s.replace(
        "_target_: egomimic.pipeline.algo.PipelineAlgo\n",
        "_target_: egomimic.pipeline.algo.PipelineAlgo\n  init_ckpt: %s\n" % SNAP.format(E=epoch),
        1)
    n = s.count("ratio_loss_weight: 1.0")
    assert n == 2, f"expected 2 ratio lines, got {n}"
    repl = f"ratio_loss_weight: {ratio_w}"
    if ste_gain != 1.0:
        repl += f"\n      ste_gain: {ste_gain}"
    s = s.replace("ratio_loss_weight: 1.0", repl)
    open(f"egomimic/hydra_configs/model/{name}.yaml", "w").write(s)
    print(name, "epoch", epoch, "ratio", ratio_w, "ste_gain", ste_gain)

make("bf_prdec_regoff499", 499, 0.0, 1.0)
make("bf_prdec_regoff999", 999, 0.0, 1.0)
make("bf_prdec_ste4_499", 499, 1.0, 4.0)
make("bf_prdec_regoff_ste4_499", 499, 0.0, 4.0)
