F="egomimic/rldb/zarr/zarr_dataset_inmem.py"
s=open(F).read()

# 1) build valid-index filter at end of __init__
old_init='''        self._mem: dict[str, np.ndarray] = {}
        self._preload()'''
new_init='''        self._mem: dict[str, np.ndarray] = {}
        self._preload()
        # Optional idle-tail / (0,0)-garbage filter (EGOVERSE_TRIM_IDLE=1).
        # The pushshapes demos record the cursor at (0,0) when it parks/leaves
        # the screen (mostly an idle tail at episode end). Those frames teach a
        # BC policy to "drive to the corner and stop". Drop them so the policy
        # only sees real pushing actions; this also cleans the quantile-norm
        # floor (computed from this same dataset).
        self._valid_indices = None
        if os.environ.get("EGOVERSE_TRIM_IDLE") == "1":
            self._build_valid_indices()'''
assert old_init in s; s=s.replace(old_init,new_init,1)

# 2) add _build_valid_indices + __len__ right after _preload definition.
# Anchor: the line that ends _preload (its last statement stores a float array).
anchor='''                self._mem[k] = np.asarray(raw, dtype=np.float32)'''
addition=anchor+'''

    def _build_valid_indices(self):
        """Drop (0,0)-garbage action frames and truncate the trailing idle tail.

        Sets self._valid_indices (sampled positions) and self.total_frames to
        the idle-tail start (so action-chunk windows never read tail garbage).
        """
        act = self._mem.get("actions")
        if act is None:
            return
        a = np.asarray(act, dtype=np.float32).reshape(len(act), -1)
        zero = (np.abs(a[:, 0]) < 1.0) & (np.abs(a[:, 1]) < 1.0)
        nz = np.where(~zero)[0]
        if len(nz) == 0:
            return  # degenerate episode: keep as-is
        tail_start = int(nz[-1]) + 1            # exclude trailing idle (0,0) run
        self.total_frames = tail_start          # clamp windows to real frames
        # sample only non-(0,0) frames before the tail (also drops scattered ones)
        self._valid_indices = [i for i in range(tail_start) if not zero[i]]
        if not self._valid_indices:             # safety: never empty
            self._valid_indices = list(range(tail_start))

    def __len__(self):
        if getattr(self, "_valid_indices", None) is not None:
            return len(self._valid_indices)
        return self.total_frames'''
assert anchor in s; s=s.replace(anchor,addition,1)

# 3) remap idx at the top of __getitem__
old_gi='''    def __getitem__(self, idx, _fallback_origin=None, _attempts=None):
        data: dict = {}'''
new_gi='''    def __getitem__(self, idx, _fallback_origin=None, _attempts=None):
        if getattr(self, "_valid_indices", None) is not None:
            idx = self._valid_indices[idx]
        data: dict = {}'''
assert old_gi in s; s=s.replace(old_gi,new_gi,1)

open(F,"w").write(s)
print("trim-idle filter added to", F)
