import numpy as np
from collections import Iterable

from itertools import combinations, chain

from abstract_context import AbstractConcept, AbstractContext
from utils_ import get_not_none, repr_set


class PatternStructure(AbstractConcept):
    def __init__(self, extent, intent, idx=None, title=None,
                 metrics=None, extent_short=None, intent_short=None,
                 is_monotonic=False, cat_feats=None,
                 ):
        assert intent is None or type(intent) == dict, 'Pattern Structure intent should be of type dict or None'
        super().__init__(extent, intent, idx, title, metrics, extent_short, intent_short, is_monotonic)
        self._cat_feats = cat_feats

    @staticmethod
    def _get_intent_as_array(int_):
        return [(f"{k} in ["+', '.join([str(v_) if type(v_) != str else f'"{v_}"' for v_ in v]) +"]") \
                    if isinstance(v, Iterable) and type(v) != str else\
                    f"{k} = {v}" for k, v in int_.items()] if int_ is not None else []

    def is_subconcept_of(self, c, trust_mode=True):
        """if a is subconcept of b, a<=b"""
        assert self._is_monotonic == c._is_monotonic, 'Cannot compare monotonic and antimonotonic concepts'
        assert type(c) == PatternStructure, "Pattern Structures can be compared only with Pattern Structures"

        if not trust_mode:
            try:
                if self._intent_short == c._intent_short and self._extent_short == c._extent_short:
                    return False
            except Exception as e:
                raise Exception(f'Cannot compare PatStructs: {self}, {c}\n{e}')

        if self._is_monotonic:
            if trust_mode:
                if not all([g in self._extent_short for g in c._extent_short]):
                    return False
                return True
            else:
                raise NotImplementedError
        else:
            if not all([g in c._extent_short for g in self._extent_short]):
                return False

            if trust_mode:
                return True  # because extent is already smaller

            for k1, v1 in c._intent_short.items():
                if v1 is None:
                    continue

                if k1 not in self._intent_short:
                    return False

                if v1 in self._cat_feats:
                    if not all([v in c._intent_short[k1] for v_ in v1]):
                        return False
                else:
                    v = self._intent_short[k1]
                    if isinstance(v1, Iterable) and type(v1) != str:
                        if isinstance(v, Iterable) and type(v) != str:
                            # v, v1 - iterables
                            if not( v1[0] <= v[0] and v[1] <= v1[1] ):
                                return False
                        else:
                            # v1 - iterable, v not iterable
                            if not (v1[0]<=v and v <= v[1]):
                                return False
                    else:
                        # v1 is not iterable
                        if isinstance(v, Iterable) and type(v) != str:
                            return False

                        if v != v1:
                            return False
                    return True

    def pretty_repr(self, print_low_neighbs=False, print_up_neighbs=False, print_level=False, metrics_to_print=None,
                    set_limit=5):
        metrics_to_print = metrics_to_print if metrics_to_print!='all' else self._metrics.keys()

        s = self._repr_concept_header(print_level)

        for t in [(self._extent, 'extent', True),
                  (self._get_intent_as_array(self._intent), 'intent', True),
                  (self._new_objs, 'new extent', True),
                  (self._new_attrs, 'new_intent', True),
                  (self._low_neighbs, 'lower neighbours', print_low_neighbs),
                  (self._up_neighbs, 'upper neighbours', print_up_neighbs), ]:
            set_, set_name, flg = t
            if not flg:
                continue
            set_ = {str(x).replace('__is__', '=').replace('__not__', '!=').replace('__lt__', '<').replace('__geq__', '>=')
                    for x in set_}
            s += repr_set(set_, set_name, lim=set_limit)

        for k in metrics_to_print:
            s += f"metric {k} = {self._metrics.get(k, 'Undefined')}\n"

        return s


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
        if pattern is None:
            return []

        assert type(pattern) == dict, "Pattern should be of type dict: attr_name->(values_interval)"
        pattern = {str(k):v for k,v in pattern.items()}
        #ms_names = [str(x) for x in pattern.keys()]
        ms_names = [x for x in pattern.keys()]
        ms_idxs = self._get_ids_in_array(ms_names, self._attrs, 'attributes') if not trust_mode else ms_names
        ms_idxs = [int(x) for x in ms_idxs]

        ext = np.arange(len(self._objs))
        for idx, m_id in enumerate(ms_idxs):
            v = pattern[ ms_names[idx]]
            if isinstance(v, Iterable) and type(v)!=str:
                if len(v)==0:
                    v = None
            if v is None:
                continue
            if m_id in self._cat_attrs_idxs:
                assert type(v) in [list, tuple,  str], 'Values of Categorical attribute should be either list or str'
                v = [v] if type(v) == str else v
                ext = ext[np.isin(self._data[ext, m_id], v)]
            else:
                #print('m_id:', type(m_id), 'v', v)
                #print(self._cat_attrs_idxs)
                number_types = (int, float, np.int64, np.int32)
                v = sorted([v, v]) if isinstance(v, number_types) else v
                assert isinstance(v, number_types) or len(v) == 2, f'Values of Real Valued attribute should be either int, float or tuple of len 2 (got {v} of type({type(v)}) feature {m_id}'
                ext = ext[ (self._data[ext, m_id] >= v[0]) & (self._data[ext, m_id] <= v[1]) ]

        ext = [str(self._objs[g]) for g in ext] if verb else list(ext)
        return ext

    def get_intent(self, gs, trust_mode=False, verb=True, return_none=False):
        gs_idxs = self._get_ids_in_array(gs, self._objs, 'objects') if not trust_mode else gs
        if len(gs) == 0:
            #pattern = {k: None for k in self._attrs}
            #pattern = {k: [] if idx in self._cat_attrs_idxs else () for idx, k in enumerate(self._attrs)}
            pattern = None
            return pattern


        pattern = {}
        for m_id, m in enumerate(self._attrs):
            k = m if verb else m_id
            if m_id in self._cat_attrs_idxs:
                v = tuple(np.unique(self._data[gs_idxs, m_id]))
                #if len(v) == len(np.unique(self._data[:, m_id])):
                #    v = None
                #elif len(v) == 1:
                #    v = v[0]
                if len(v) == 1:
                    v = v[0]
            else:
                v = self._data[gs_idxs, m_id]
                v = None if any([v_ is None or np.isnan(v_) for v_ in v]) or len(v)==0 else v
                if v is not None:
                    v = (min(v), max(v)) if isinstance(v, Iterable) and type(v)!=str else (v, v)
                    #if v[0] == self._data[:, m_id].min() and v[1] == self._data[:, m_id].max():
                    #    v = None
                    #elif v[0] == v[1]:
                    #    v = v[0]
                    if v[0] == v[1]:
                        v = v[0]
            pattern[k] = v

        pattern = {k: v for k, v in pattern.items() if v is not None} if not return_none else pattern
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
