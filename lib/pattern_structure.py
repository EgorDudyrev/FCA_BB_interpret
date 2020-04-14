import numpy as np

from itertools import combinations, chain

from abstract_context import AbstractContext

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


class MultiValuedContext(AbstractContext):
    def __init__(self, data, objs=None, attrs=None, y_true=None, y_pred=None,  cat_attrs=None):
        super().__init__(data, objs, attrs, y_true, y_pred)
        data, objs, attrs, y_true, y_pred = super()._check_input_data(data, objs, attrs, y_true, y_pred)

        self._data_full = data
        self._objs_full = objs
        self._attrs_full = attrs

        self._attrs = attrs
        self._objs = objs
        self._data = data

        self.cat_attrs = cat_attrs
        self._cat_attrs_idxs = [idx for idx, m in enumerate(attrs) if m in cat_attrs]

    def _reduce_context(self, data, objs, attrs):
        raise NotImplementedError

    def get_same_objs(self, g):
        raise NotImplementedError

    def get_same_attrs(self, m):
        raise NotImplementedError

    def get_extent(self, pattern, trust_mode=False, verb=True):
        assert type(pattern) == dict, "Pattern should be of type dict: attr_name->(values_interval)"
        ms_names = [str(x) for x in pattern.keys()]
        ms_idxs = self._get_ids_in_array(ms_names, self._attrs, 'attributes') if not trust_mode else ms_names

        ext = np.arange(len(self._objs))
        for idx, m_id in enumerate(ms_idxs):
            v = pattern[ ms_names[idx]]
            if v is None:
                continue
            if m_id in self._cat_attrs_idxs:
                assert type(v) in [list, tuple,  str], 'Values of Categorical attribute should be either list or str'
                v = [v] if type(v) == str else v
                ext = ext[np.isin(self._data[ext, m_id], v)]
            else:
                v = [v, v] if type(v) in [float, int] else v
                v = sorted(v)
                assert len(v), 'Values of Real Valued attribute should be either int, float or tuple of len 2'
                ext = ext[ (self._data[ext, m_id] >= v[0]) & (self._data[ext, m_id] <= v[1]) ]

        ext = [self._objs[g] for g in ext] if verb else ext
        return ext

    def get_intent(self, gs, trust_mode=False, verb=True, return_none = False):
        gs_idxs = self._get_ids_in_array(gs, self._objs, 'objects') if not trust_mode else gs

        pattern = {}
        for m_id, m in enumerate(self._attrs):
            k = m if verb else m_id
            if m_id in self._cat_attrs_idxs:
                v = tuple(np.unique(self._data[gs_idxs, m_id]))
                if len(v) == len(np.unique(self._data[:, m_id])):
                    v = None
                elif len(v) == 1:
                    v = v[0]
            else:
                v = self._data[gs_idxs, m_id]
                v = None if any([v_ is None or np.isnan(v_) for v_ in v]) else v
                if v is not None:
                    v = (v.min(), v.max())
                    if v[0] == self._data[:, m_id].min() and v[1] == self._data[:, m_id].max():
                        v = None
                    elif v[0] == v[1]:
                        v = v[0]
            pattern[k] = v

        pattern = {k:v for k, v in pattern.items() if v is not None} if not return_none else pattern
        return pattern

    def __repr__(self):
        s = f"Num of objects: {len(self._objs)}, Num of attrs: {len(self._attrs)}\n"
        s += repr_set(self._objs, 'Objects', True, lim=5)
        s += repr_set(self._attrs, 'Attrs', True, lim=5)
        #s += self.get_table().head().__repr__()
        return s

    def calc_implications(self, max_len=None, use_tqdm=False):
        raise NotImplementedError

    def repr_implications(self, impls=None, max_len=None, use_tqdm=False):
        raise NotImplementedError

    def save_implications(self, impls=None, max_len=None, use_tqdm=False):
        raise NotImplementedError
