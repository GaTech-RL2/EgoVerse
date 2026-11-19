"""Expose native float32 numbers as short decimals with identical execution bits."""
import copy
import math
import struct

VERSION='native-float32-short-decimal-v1'


def bits(value):return struct.pack('!f',value)


def public_number(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Native numbers must be finite')
    original=bits(value)
    if struct.unpack('!f',original)[0]!=value:
        raise ValueError('A native float64 value cannot be relabeled as float32')
    for digits in range(1,10):
        candidate=float(format(value,f'.{digits}g'))
        if bits(candidate)==original:return candidate
    raise ValueError('Native float32 has no round-trip decimal')


def public_contract(contract):
    result=copy.deepcopy(contract)
    for entry in result['action_types'].values():
        for key in ['lower','upper']:entry[key]=[public_number(v)for v in entry[key]]
    result['numeric_representation']=VERSION
    return result


def verify_contract(actual,original):
    if actual.get('numeric_representation')!=VERSION:raise ValueError('Native decimal fix is not active')
    if set(actual['action_types'])!=set(original['action_types']):raise ValueError('Native action types changed')
    for name,entry in original['action_types'].items():
        candidate=actual['action_types'][name]
        for key in ['dim','layout']:
            if candidate[key]!=entry[key]:raise ValueError('Native action layout changed')
        for key in ['lower','upper']:
            if len(candidate[key])!=len(entry[key])or any(bits(a)!=bits(b)for a,b in zip(candidate[key],entry[key])):
                raise ValueError('Native controller bounds changed')
