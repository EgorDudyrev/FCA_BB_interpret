import numpy as np
import pandas as pd
from tqdm import tqdm

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


class Concept:
    def __init__(self, extent, intent, idx=None, pattern=None, title=None,
                 y_true_mean=None, y_pred_mean=None, metrics=None, extent_short=None, intent_short=None,
                 is_monotonic=False):
        self._extent = np.array(extent)
        self._intent = np.array(intent)
        self._extent_short = get_not_none(extent_short, self._extent)
        self._intent_short = get_not_none(intent_short, self._intent)
        self._idx = idx
        self._title = title
        self._low_neighbs = None
        self._up_neighbs = None
        self._level = None
        self._new_objs = None
        self._new_attrs = None
        self._metrics = get_not_none(metrics, {})
        self._is_monotonic = is_monotonic

    def get_extent(self):
        return self._extent

    def get_intent(self):
        return self._intent

    def get_id(self):
        return self._idx

    def get_level(self):
        return self._level

    def get_lower_neighbs(self):
        return self._low_neighbs

    def get_upper_neighbs(self):
        return self._up_neighbs

    def _repr_concept_header(self, print_level=True):
        s = f"Concept"
        s += f" {self._idx}" if self._idx is not None else ''
        s += f" {self._title}" if self._title is not None else ''
        s += f"\n"
        s += f"level: {self._level}" if print_level and self._level is not None else ''
        s += '\n'
        return s

    def __repr__(self):
        s = self._repr_concept_header()

        for set_, set_name in [(self._extent, 'extent'),
                               (self._intent, 'intent'),
                               (self._new_objs, 'new extent'),
                               (self._new_attrs, 'new_intent'),
                               (self._low_neighbs, 'lower neighbours'),
                               (self._up_neighbs, 'upper neighbours'), ]:
            s += repr_set(set_, set_name)

        for k, v in self._metrics.items():
            s += f'metric {k} = {v}\n'

        return s

    def pretty_repr(self, print_low_neighbs=False, print_up_neighbs=False, print_level=False, metrics_to_print=None,
                    set_limit=5):
        metrics_to_print = metrics_to_print if metrics_to_print!='all' else self._metrics.keys()

        s = self._repr_concept_header(print_level)

        for t in [(self._extent, 'extent', True),
                  (self._intent, 'intent', True),
                  (self._new_objs, 'new extent', True),
                  (self._new_attrs, 'new_intent', True),
                  (self._low_neighbs, 'lower neighbours', print_low_neighbs),
                  (self._up_neighbs, 'upper neighbours', print_up_neighbs), ]:
            set_, set_name, flg = t
            if not flg:
                continue
            set_ = {x.replace('__is__', '=').replace('__not__', '!=').replace('__lt__', '<').replace('__geq__', '>=')
                    for x in set_}
            s += repr_set(set_, set_name, lim=set_limit)

        for k in metrics_to_print:
            s += f"metric {k} = {self._metrics.get(k, 'Undefined')}\n"

        return s

    def __str__(self):
        s = self._repr_concept_header()
        s += f"({len(self._extent)} objs, {len(self._intent)} attrs)"
        s = s.replace('\n', ' ')
        return s

    def is_subconcept_of(self, c):
        """if a is subconcept of b, a<=b"""
        assert self._is_monotonic == c._is_monotonic, 'Cannot compare monotonic and antimonotonic concepts'
        if self._is_monotonic:
            return all([g in self._extent_short for g in c._extent_short]) \
                   and all([m in self._intent_short for m in c._intent_short])
        else:
            return all([g in c._extent_short for g in self._extent_short]) \
                   and all([m in self._intent_short for m in c._intent_short])

class BinaryContext(AbstractContext):
    def __init__(self, data, objs=None, attrs=None, y_true=None, y_pred=None, ):  # cat_attrs=None):
        super().__init__(data, objs, attrs, y_true, y_pred)
        data, objs, attrs, y_true, y_pred = super()._check_input_data(data, objs, attrs, y_true, y_pred)

        self._data_full = data
        self._objs_full = objs
        self._attrs_full = attrs
        same_objs, same_attrs = self._reduce_context(data, objs, attrs)

        self._same_objs = same_objs
        self._same_attrs = same_attrs
        self._data = data[[idx for idx, g in enumerate(objs) if g in same_objs.keys()], :] \
                         [:,[idx for idx, m in enumerate(attrs) if m in same_attrs.keys()]]
        self._objs = np.array([g for idx, g in enumerate(objs) if g in same_objs.keys()])
        self._attrs = np.array([m for idx, m in enumerate(attrs) if m in same_attrs.keys()])

    def _reduce_context(self, data, objs, attrs):
        same_attrs = {}
        saw_attrs = set()
        for i in range(data.shape[1]):
            if i in saw_attrs:
                continue
            idxs = np.arange(i + 1, data.shape[1])[(data.T[i] == data.T[i + 1:]).mean(1) == 1]
            idxs = [idx for idx in idxs if idx not in saw_attrs]
            same_attrs[i] = idxs
            for idx in idxs:
                saw_attrs.add(idx)

        same_objs = {}
        saw_objs = set()
        for i in range(data.shape[0]):
            if i in saw_objs:
                continue
            idxs = np.arange(i + 1, data.shape[0])[(data[i] == data[i + 1:]).mean(1) == 1]
            idxs = [idx for idx in idxs if idx not in saw_objs]
            same_objs[i] = idxs
            for idx in idxs:
                saw_objs.add(idx)

        same_attrs = {attrs[k]: np.array(attrs)[v] if len(v) > 0 else v for k, v in same_attrs.items()}
        same_objs = {objs[k]: np.array(objs)[v] if len(v) > 0 else v for k, v in same_objs.items()}

        return same_objs, same_attrs

    def get_same_objs(self, g):
        if g in self._same_objs.keys():
            return self._same_objs[g]
        for k, v in self._same_objs.items():
            if g in v:
                return [k]+[x for x in v if x != g]

    def get_same_attrs(self, m):
        if m in self._same_attrs.keys():
            return self._same_attrs[m]
        for k, v in self._same_attrs.items():
            if m in v:
                return [k]+[x for x in v if x != m]

    def get_extent(self, ms, trust_mode=False, verb=True):
        ms_idxs = self._get_ids_in_array(ms, self._attrs, 'attributes') if not trust_mode else ms

        ext = np.arange(len(self._objs))
        ext = list(ext[self._data[:, ms_idxs].sum(1) == len(ms_idxs)])
        ext = [self._objs[g] for g in ext] if verb else ext
        return ext

    def get_intent(self, gs, trust_mode=False, verb=True):
        gs_idxs = self._get_ids_in_array(gs, self._objs, 'objects') if not trust_mode else gs

        int_ = np.arange(len(self._attrs))
        cntx = self._data[gs_idxs]
        int_ = list(int_[cntx.sum(0) == len(gs_idxs)])
        int_ = [self._attrs[m] for m in int_] if verb else int_
        return int_

    def __repr__(self):
        s = f"Num of objects: {len(self._objs)}, Num of attrs: {len(self._attrs)}\n"
        s += repr_set(self._objs, 'Objects', True, lim=5)
        s += repr_set(self._attrs, 'Attrs', True, lim=5)
        #s += self.get_table().head().__repr__()
        return s

    def calc_implications(self, max_len=None, use_tqdm=False):
        max_len = get_not_none(max_len, len(self._attrs)-1)

        impls = {f: set() for f in self._attrs }
        for idx, f in tqdm(enumerate(self._attrs), disable=not use_tqdm):
            ext_ = self.get_extent(idx)
            for comb in powerset([x for x in range(len(self._attrs)) if x!=idx], max_len):
                for comb1 in powerset(comb):
                    if tuple(comb1) in impls[f]:
                        break
                else:
                    ext_1 = self.get_extent(comb, trust_mode=True)
                    if all(np.isin(ext_1, ext_)) and len(ext_1)>0:
                        impls[f].add(tuple(comb))
        impls = {k: set([tuple(self._attrs[list(v)]) for v in vs]) for k, vs in impls.items()}
        impls_verb = {}
        for k,vs in impls.items():
            for v in vs:
                impls_verb[v] = impls_verb.get(v, set()) | set([k])
        return impls_verb

    def repr_implications(self, impls=None, max_len=None, use_tqdm=False):
        if impls is None:
            impls = self.calc_implications(max_len, use_tqdm)

        s = ""
        for k, vs in sorted([(k, vs) for k, vs in impls.items()], key=lambda x: (len(x[0]), ','.join(x[0]))):
            for v in vs:
              s += f"{', '.join(k)} => {v}\n"
        return s

    def save_implications(self, impls=None, max_len=None, use_tqdm=False):
        if impls is None:
            impls = self.calc_implications(max_len, use_tqdm)

        self._impls = impls

class Binarizer:
        def binarize_ds(self, ds, cat_feats, thresholds, cases):
            bin_ds = pd.DataFrame()
            for f in cat_feats:
                if ds[f].nunique() == 2:
                    if ds[f].dtype == bool:
                        bin_ds[f"{f}__is__True"] = ds[f]
                    elif all(ds[f].unique() == [0, 1]):
                        bin_ds[f"{f}__is__True"] = ds[f].astype(bool)
                    bin_ds[f"{f}__not__True"] = ~ds[f]
                else:
                    for v in ds[f].unique():
                        bin_ds[f"{f}__is__{v}"] = ds[f] == v
                        bin_ds[f"{f}__not__{v}"] = ~bin_ds[f"{f}__is__{v}"]

            feat_orders = []
            for f, ts in thresholds.items():
                fo_geq = []
                fo_le = []
                for t in sorted(ts):
                    bin_ds[f"{f}__geq__{t}"] = ds[f] >= t
                    bin_ds[f"{f}__lt__{t}"] = ds[f] < t
                    fo_geq = [f"{f}__geq__{t}"] + fo_geq
                    fo_le = fo_le + [f"{f}__lt__{t}"]
                feat_orders += [tuple(fo_geq), tuple(fo_le)]

            for f, cs in cases.items():
                for c in cs:
                    bin_ds[f"{f}__is__{c}"] = ds[f] == c
                    bin_ds[f"{f}__not__{c}"] = ~bin_ds[f"{f}__is__{c}"]

            bin_ds = bin_ds.astype(bool)
            return bin_ds, feat_orders

        def test_feats(self, bin_ds, fs, y, metric, n_estimators=100):
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if set(y) == {0, 1}:
                rf = RandomForestClassifier(n_estimators=n_estimators)
            else:
                rf = RandomForestRegressor(n_estimators=n_estimators)

            rf.fit(bin_ds[fs], y)
            m = metric(y, rf.predict(bin_ds[fs]))
            s = pd.Series(rf.feature_importances_, index=fs).sort_values(ascending=False)
            return m, s

        def squeeze_bin_dataset(self, bin_ds, y, metric, metric_lim, use_tqdm=False, n_estimators=100,
                                min_n_feats=None):
            fs = bin_ds.columns
            max_metric = self.test_feats(bin_ds, fs, y, metric, n_estimators)[0]
            assert max_metric >= metric_lim, f'Target metric limit is unreachable (max is {max_metric})'

            for i in tqdm(range(len(bin_ds.columns)), disable=not use_tqdm):
                if min_n_feats is not None and len(fs) <= min_n_feats:
                    break

                for f in fs[::-1]:
                    fs_ = [f_ for f_ in fs if f_ != f]
                    ac, s = self.test_feats(bin_ds, fs_, y, metric, n_estimators)

                    if ac >= metric_lim:
                        fs = list(s.index)
                        break
                else:
                    break
            return fs
