from itertools import combinations, chain


def get_not_none(v, v_if_none):
    return v if v is not None else v_if_none


def repr_set(set_, set_name, to_new_line=True, lim=None):
    if set_ is None:
        return ''
    lim = get_not_none(lim, len(set_))
    rpr = f"{set_name} (len: {len(set_)}): "
    rpr += f"{(', '.join(f'{v}' for v in list(set_)[:lim])+(',...' if len(set_)>lim else '')) if len(set_) > 0 else '∅'}"
    rpr += '\n' if to_new_line else ''
    return rpr


def powerset(iterable, max_len=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    max_len = get_not_none(max_len, len(s))
    return chain.from_iterable(combinations(s, r) for r in range(max_len + 1))