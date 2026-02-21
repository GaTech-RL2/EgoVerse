import torch, pprint

def summarize(path):
    ckpt = torch.load(path, map_location="cpu")
    print("\n===", path, "===")
    print("top-level keys:", sorted(ckpt.keys()))
    print("pl version:", ckpt.get("pytorch-lightning_version") or ckpt.get("lightning_version"))
    hp = ckpt.get("hyper_parameters", {})
    print("hyper_parameters keys count:", len(hp))
    print("has robomimic_model in hyper_parameters?:", "robomimic_model" in hp)
    # show likely constructor-related keys
    print("hp_keys:", sorted(hp.keys()))
    interesting = [k for k in hp.keys() if any(s in k.lower() for s in ["model", "policy", "config", "algo", "robomimic"])]
    print("interesting hparams keys:", sorted(interesting)[:50])
    # optional: show hparams content types
    for k in sorted(interesting)[:20]:
        v = hp[k]
        print(f"  {k}: {type(v)}")

summarize("/home/robot/robot_ws/egomimic/robot/models/hpt_objcont_bc/0/checkpoints/epoch_epoch=1599.ckpt")
summarize("/home/robot/robot_ws/egomimic/robot/models/pi_eva_objcon_full_folder/0/checkpoints/epoch_epoch=399.ckpt")
