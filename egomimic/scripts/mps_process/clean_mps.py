import os
from cloudpathlib import S3Path
from egomimic.utils.aws.aws_data_utils import load_env, get_cloudpathlib_s3_client

def clean_mps_remote(raw_remote_prefix: str):
    load_env()
    cloudpathlib_s3 = get_cloudpathlib_s3_client()

    root = S3Path(raw_remote_prefix, client=cloudpathlib_s3)
    try:
        walker = root.walk(topdown=True)
        can_prune = True
    except TypeError:
        walker = root.walk()
        can_prune = False

    delete_paths = []
    for dirpath, dirnames, filenames in walker:
        try:
            rel = dirpath.relative_to(root)
            rel_str = rel.as_posix()
            rel_parts = () if rel_str in (".", "") else rel.parts
        except Exception:
            rel_parts = ()

        depth = len(rel_parts)

        if depth == 0:
            if can_prune:
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d.startswith("mps_") and d.endswith("_vrs")
                ]

        elif depth == 1:
            d0 = rel_parts[0]
            if d0.startswith("mps_") and d0.endswith("_vrs"):
                name = d0[len("mps_") : -len("_vrs")]
                has_hand = "hand_tracking" in dirnames
                has_slam = "slam" in dirnames
                has_gaze = "eye_gaze" in dirnames
                if not (has_hand and has_slam and has_gaze):
                    delete_paths.append(dirpath)
            if can_prune:
                dirnames[:] = []

        else:
            if can_prune:
                dirnames[:] = []
    for path in delete_paths:
        print(f"Deleting {path.name}")
        path.rmtree()

if __name__ == "__main__":
    raw_remote_prefix = os.environ.get(
            "RAW_REMOTE_PREFIX", "s3://rldb/raw_v2/aria"
        ).rstrip("/")
    clean_mps_remote(raw_remote_prefix)