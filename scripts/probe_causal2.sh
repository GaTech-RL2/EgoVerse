#!/bin/bash
set -uxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7

# Bypass activate (broken), call .venv/bin/python directly.
.venv/bin/python <<'PY'
import sys
print('python:', sys.executable)
print('site-packages relevant:', [p for p in sys.path if 'EgoVerse' in p])

import causal_conv1d
print('causal_conv1d __file__:', causal_conv1d.__file__)

# Now the C extension itself
try:
    import causal_conv1d_cuda
    print('CUDA ext OK at', causal_conv1d_cuda.__file__)
    print('exported:', [x for x in dir(causal_conv1d_cuda) if not x.startswith('_')][:10])
except Exception as e:
    print('CUDA ext IMPORT ERROR:', type(e).__name__, e)

# How mamba_ssm tries to import it
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print('causal_conv1d_fn:', causal_conv1d_fn)
except Exception as e:
    print('IMPORT _fn ERROR:', type(e).__name__, e)

# The actual symbol that came back None
try:
    from mamba_ssm.ops.triton.ssd_combined import causal_conv1d_fwd_function
    print('causal_conv1d_fwd_function:', causal_conv1d_fwd_function)
except Exception as e:
    print('IMPORT fwd_function ERROR:', type(e).__name__, e)
PY
