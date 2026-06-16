#!/bin/bash
set -uxo pipefail
cd /storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse7
source .venv/bin/activate

python -c "
import sys
print('python:', sys.executable)
print('site-packages:', [p for p in sys.path if 'EgoVerse7' in p])
import causal_conv1d
print('causal_conv1d module:', causal_conv1d.__file__)
print('causal_conv1d version:', getattr(causal_conv1d, '__version__', '?'))
print('dir(causal_conv1d):', dir(causal_conv1d))
try:
    import causal_conv1d_cuda
    print('causal_conv1d_cuda OK:', causal_conv1d_cuda.__file__)
    print('dir(causal_conv1d_cuda):', [x for x in dir(causal_conv1d_cuda) if not x.startswith('_')])
except Exception as e:
    print('causal_conv1d_cuda IMPORT ERROR:', repr(e))
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print('causal_conv1d_fn:', causal_conv1d_fn)
except Exception as e:
    print('causal_conv1d_fn IMPORT ERROR:', repr(e))
"
