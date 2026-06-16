def keep_keys(d, to_keep):
    for k in to_keep:
        assert k in d.keys(), f"key {k} not in dict"

    to_delete = [k for k in d.keys() if k not in to_keep]

    for k in to_delete:
        del d[k]

    return d
