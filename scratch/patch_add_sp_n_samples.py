p = "egomimic/algo/dfot/algo.py"
s = open(p).read()
old1 = "        sp_commit: int = 1,"
new1 = "        sp_commit: int = 1,\n        sp_n_samples: int = 1,"
if "sp_n_samples: int = 1," not in s:
    assert old1 in s, "sp_commit arg not found"
    s = s.replace(old1, new1, 1)
old2 = "        self.sp_commit = int(sp_commit)"
new2 = "        self.sp_commit = int(sp_commit)\n        self.sp_n_samples = int(sp_n_samples)"
if "self.sp_n_samples = int(sp_n_samples)" not in s:
    assert old2 in s, "self.sp_commit assignment not found"
    s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("ADDED sp_n_samples to DFoT.__init__")
