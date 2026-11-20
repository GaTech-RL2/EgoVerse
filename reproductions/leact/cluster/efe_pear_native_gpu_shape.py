"""Strict opt-in GPU ownership for smaller privileged native EFE workers.

The NVIDIA device plugin's envvar/UUID strategy advertises allocated devices in
NVIDIA_VISIBLE_DEVICES. A privileged pod may enumerate additional host GPUs, so
enumeration order or a matching device count never proves ownership here.
"""
import os
import re
import subprocess


VERSION = 'native-efe-assigned-uuid-v1'
PLUGIN_DOC = 'https://github.com/NVIDIA/k8s-device-plugin#configuration-option-details'
UUID = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')


def assigned_devices(count, *, environ=None, query=None):
    if type(count) is not int or count not in {2, 4}:
        raise ValueError('Smaller native EFE ownership requires an explicit 2- or 4-GPU shape')
    environ = os.environ if environ is None else environ
    raw = environ.get('NVIDIA_VISIBLE_DEVICES', '')
    devices = [value.strip() for value in raw.split(',')]
    if (len(devices) != count or len(set(devices)) != count
            or any(UUID.fullmatch(value) is None for value in devices)):
        raise ValueError('Cannot prove allocated GPU UUIDs: require scheduler-provided '
                         'NVIDIA_VISIBLE_DEVICES containing exactly the requested full GPU UUIDs; '
                         'all, indices, MIG IDs and missing/duplicate assignments are ambiguous')
    if query is None:
        query = lambda: subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=30).splitlines()
    observed = [value.strip() for value in query()]
    if (not observed or len(set(observed)) != len(observed)
            or any(UUID.fullmatch(value) is None for value in observed)
            or not set(devices).issubset(observed)):
        raise ValueError('Allocated GPU UUIDs do not match unambiguous native GPU enumeration')
    return devices, {'version': VERSION, 'requested_gpu_count': count,
                     'assignment_source': 'Inherited NVIDIA_VISIBLE_DEVICES full UUID list',
                     'assigned_gpu_uuids': devices, 'enumerated_gpu_uuids': observed,
                     'selection_rule': 'Exact assigned UUIDs; never the first N host GPUs',
                     'documentation': PLUGIN_DOC}


def verify_cdi(devices, receipt):
    names = {line.strip() for line in receipt.get('devices', [])}
    required = {'nvidia.com/gpu='+device for device in devices}
    if not required.issubset(names):
        raise ValueError('Generated CDI specification lacks an exact assigned GPU UUID device')
    return sorted(required)
