import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import concepts as concepts_mit
import networkx as nx
import plotly.graph_objects as go

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

        for k, v in self._metrics:
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


class Context:
    def __init__(self, data, objs=None, attrs=None, y_true=None, y_pred=None, ):  # cat_attrs=None):
        if type(data) == list:
            data = np.array(data)

        if type(data) == pd.DataFrame:
            objs = list(data.index) if objs is None else objs
            attrs = list(data.columns) if attrs is None else attrs
            data = data.values
        elif type(data) == np.ndarray:
            objs = list(range(len(data.shape[1]))) if objs is None else objs
            attrs = list(range(len(objs[0]))) if attrs is None else attrs
        else:
            raise TypeError(f"DataType {type(data)} is not understood. np.ndarray or pandas.DataFrame is required")

        objs = np.array([str(g) for g in objs])
        attrs = np.array([str(m) for m in attrs])

        assert data.dtype == bool, 'Only Boolean contexts are supported for now'

        if y_true is not None:
            if type(y_true) == pd.Series:
                self._y_true = y_true.values
            elif type(y_true) == np.ndarray:
                self._y_true = y_true
            else:
                raise TypeError(f"DataType {type(y_true)} is not understood. np.ndarray or pandas.Series is required")
            assert len(y_true) == len(
                data), f'Data and Y_vals have different num of objects ( Data: {len(data)}, y_vals: {len(y_true)})'
        else:
            self._y_true = None

        if y_pred is not None:
            if type(y_pred) == pd.Series:
                self._y_pred = y_pred.values
            elif type(y_pred) == np.ndarray:
                self._y_pred = y_pred
            else:
                raise TypeError(f"DataType {type(y_pred)} is not understood. np.ndarray or pandas.Series is required")
            assert len(y_pred) == len(data), f'Data and Y_vals have different num of objects ( Data: {len(data)}, y_vals: {len(y_pred)})'
        else:
            self._y_pred = None

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

    def get_attrs(self, is_full=True):
        return self._attrs_full if is_full else self._attrs

    def get_objs(self, is_full=True):
        return self._objs_full if is_full else self._objs

    def get_data(self):
        return self._data_full

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

    def get_y_true(self, objs):
        if objs is None or len(objs) == 0 or self._y_true is None:
            return None
        return self._y_true[np.isin(self._objs_full, objs)]

    def get_y_pred(self, objs):
        if objs is None or len(objs) == 0 or self._y_pred is None:
            return None
        return self._y_pred[np.isin(self._objs_full, objs)]

    def _get_id_in_array(self, x, ar, ar_name):
        if type(x) == int:
            idx = x
            if idx < 0 or idx > len(ar)-1:
                raise ValueError(f"There are only {len(ar)} {ar_name} (Suggested ({idx}")
        elif type(x) == str:
            if x not in ar:
                raise ValueError(f"No such {x} in {ar_name}")
            idx = np.argmax(ar == x)
        else:
            raise TypeError(f"Possible values for {ar_name} are string and int type")
        return idx

    def _get_ids_in_array(self, xs, ar, ar_name):
        idxs = []
        error = []
        xs = list(xs) if type(xs) == tuple else [xs] if type(xs) != list else xs
        for x in xs:
            try:
                idxs.append(self._get_id_in_array(x, ar, ar_name))
            except ValueError:
                error.append(x)
        if len(error) > 0:
            raise ValueError(f"Wrong {ar_name} are given: {error}")
        return idxs

    def get_attr_values(self, m, trust_mode=False):
        m_idx = self._get_id_in_array(m, self._attrs, 'attributes') if not trust_mode else m
        return self._data[:, m_idx]

    def get_obj_values(self, g, trust_mode=False):
        g_idx = self._get_id_in_array(g, self._objs, 'objects') if not trust_mode else g
        return self._data[g_idx]

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

    def get_table(self, is_full=True):
        return pd.DataFrame(self._data, index=self._objs_full if is_full else self._objs,
                            columns=self._attrs_full if is_full else self._attrs)

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

    @staticmethod
    def binarize_ds(ds, cat_feats, thresholds, cases):
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
        return bin_ds, feat_orders


class FormalManager:
    def __init__(self, context, ds_obj=None, target_attr=None, cat_feats=None, task_type=None):
        self._context = context
        if ds_obj is not None:
            ds_obj.index = ds_obj.index.astype(str)
        self._ds_obj = ds_obj
        self._concepts = None
        self._target_attr = target_attr
        self._top_concept = None
        self._cat_feats = cat_feats
        self._task_type = task_type

    def get_context(self):
        return self._context

    def get_concepts(self):
        return self._concepts

    def get_concept_by_id(self, id_):
        cpt = [c for c in self._concepts if c.get_id() == id_]
        if len(cpt) == 0:
            return None
        return cpt[0]

    def sort_concepts(self, concepts):
        return sorted(concepts, key=lambda c: (len(c.get_intent()), ','.join(c.get_intent())))

    def construct_concepts(self, algo='mit', max_iters_num=None, max_num_attrs=None, min_num_objs=None, use_tqdm=True,
                           is_monotonic=False, stop_on_strong_hyp=False, stop_after_strong_hyp=False):
        if algo == 'CBO':
            concepts = self._close_by_one(max_iters_num, max_num_attrs, min_num_objs, use_tqdm,
                                          stop_on_strong_hyp=stop_on_strong_hyp,
                                          stop_after_strong_hyp=stop_after_strong_hyp, is_monotonic=is_monotonic)
            concepts = {Concept(tuple(self._context.get_objs(is_full=False)[c.get_extent()])
                                    if len(c.get_extent()) > 0 else tuple(),
                                tuple(self._context.get_attrs()[c.get_intent()])
                                    if len(c.get_intent()) > 0 else tuple()
                                ) for c in concepts}
        elif algo == 'mit':
            concepts = self._concepts_by_mit()
        else:
            raise ValueError('The only supported algorithm is CBO (CloseByOne) and "mit" (from library "concepts")')

        if is_monotonic:
            concepts = {Concept([g for g in self._context.get_objs(is_full=False) if g not in c.get_extent()], c.get_intent())
                        for c in concepts}

        new_concepts = set()
        for idx, c in enumerate(self.sort_concepts(concepts)):
            ext_short = c.get_extent()
            int_short = c.get_intent()
            ext_ = [g_ for g in ext_short for g_ in [g] + list(self._context.get_same_objs(g))]
            int_ = [m_ for m in int_short for m_ in [m] + list(self._context.get_same_attrs(m))]

            metrics = self._calc_metrics_inconcept(ext_) if len(ext_) > 0 else None

            new_concepts.add(Concept(ext_, int_, idx=idx,
                                     metrics=metrics,
                                     extent_short=ext_short, intent_short=int_short,
                                     is_monotonic=is_monotonic))
        concepts = new_concepts
        self._concepts = concepts

        self._top_concept = self.get_concept_by_id(0)

    def delete_concept(self, c_idx):
        c = self.get_concept_by_id(c_idx)
        upns, lns = c.get_upper_neighbs(), c.get_lower_neighbs()
        if upns is None or len(upns) == 0:
            raise KeyError(f'Cannot delete concept {c_idx}. It may be supremum')

        # if lns is None or len(lns) == 0:
        #	raise KeyError(f'Cannot delete concept {c_idx}. It may be infinum')

        if upns is not None:
            for upn_id in upns:
                upn = self.get_concept_by_id(upn_id)
                upn._low_neighbs.remove(c_idx)
                upn._low_neighbs = upn._low_neighbs | lns

        if lns is not None:
            for ln_id in lns:
                ln = self.get_concept_by_id(ln_id)
                ln._up_neighbs.remove(c_idx)
                ln._up_neighbs = ln._up_neighbs | upns

        self._concepts = [c_ for c_ in self._concepts if c_ != c]

        self._find_new_concept_objatr()

    def _concepts_by_mit(self):
        cntx_mit = concepts_mit.Context(self._context.get_objs(is_full=False),
                                        self._context.get_attrs(is_full=False),
                                        self._context._data
                                        )

        self._lattice_mit = cntx_mit.lattice
        concepts = {Concept(ext_, int_) for ext_, int_ in self._lattice_mit}
        return concepts

    def _calc_metrics_inconcept(self, ext_):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
            r2_score, mean_absolute_error, mean_squared_error

        y_true = self._context.get_y_true(ext_)
        y_pred = self._context.get_y_pred(ext_)
        if y_true is None or y_pred is None:
            return None
        if self._task_type == 'regression':
            ms = {
                'r2': r2_score(y_true, y_pred),
                'me': np.mean(y_true - y_pred),
                'ame': np.abs(np.mean(y_true - y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mape': np.mean(np.abs(y_true - y_pred) / y_true),
            }
        elif self._task_type == 'binary classification':
            ms = {
                'accuracy': round(accuracy_score(y_true, y_pred), 2),
                'precision': round(precision_score(y_true, y_pred), 2),
                'recall': round(recall_score(y_true, y_pred), 2),
                'neg_precision': round(precision_score(1 - y_true, 1 - y_pred), 2),
                'neg_recall': round(recall_score(1 - y_true, 1 - y_pred), 2),
                'f1_score': round(f1_score(y_true, y_pred), 2),
            }
        else:
            raise ValueError(
                f"Given task type {self._task_type} is not supported. Possible values are 'regression', 'binary classification'")
        ms['y_pred_mean'] = np.mean(y_pred)
        ms['y_true_mean'] = np.mean(y_true)
        return ms

    def _close_by_one(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm, stop_on_strong_hyp=False,
                      stop_after_strong_hyp=False, is_monotonic=False):
        cntx = self._context
        n_attrs = len(cntx.get_attrs())
        combs_to_check = [[]]
        concepts = set()
        if use_tqdm:
            if max_num_attrs is not None and min_num_objs is not None:
                tot = min(sum([sp.misc.comb(N=n_attrs, k=x) for x in range(0, max_num_attrs + 1)]),
                          sum([sp.misc.comb(N=n_attrs, k=x) for x in range(0, min_num_objs + 1)]))
                t = tqdm(total=tot)
            elif max_num_attrs is not None:
                t = tqdm(total=sum([sp.misc.comb(N=n_attrs, k=x) for x in range(0, max_num_attrs + 1)]))
            elif min_num_objs is not None:
                t = tqdm(total=sum([sp.misc.comb(N=n_attrs, k=x) for x in range(0, min_num_objs + 1)]))
            else:
                t = tqdm(total=len(combs_to_check))

        iter = 0
        saved_ints = set()
        while len(combs_to_check) > 0:
            iter += 1
            if max_iters_num is not None and iter >= max_iters_num:
                break
            comb = combs_to_check.pop(0)

            if max_num_attrs is not None and len(comb) > max_num_attrs:
                continue
            ext_ = cntx.get_extent(comb, trust_mode=True, verb=False)
            if min_num_objs is not None and len(ext_) <= min_num_objs:
                continue
            int_ = cntx.get_intent(ext_, trust_mode=True, verb=False)

            new_int_ = [x for x in int_ if x not in comb]

            if (len(comb) > 0 and any([x < comb[-1] for x in new_int_])) or tuple(int_) in saved_ints:
                if use_tqdm:
                    t.update()
                continue

            concepts.add(Concept(ext_, int_))
            saved_ints.add(tuple(int_))
            flg = np.isin(self._context._objs_full, ext_)
            mpred = self._context._y_pred[flg if not is_monotonic else ~flg].mean()
            if stop_on_strong_hyp and mpred in [0, 1]:
                new_combs = []
            elif stop_after_strong_hyp and mpred not in [0, 1] and len(int_) > 0:
                new_combs = []
            else:
                new_combs = [int_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_attrs) if
                             x not in int_]
            combs_to_check = new_combs + combs_to_check
            if use_tqdm:
                t.update()
                if max_num_attrs is None or min_num_objs is not None:
                    t.total += len(new_combs)

        if tuple([]) not in saved_ints:
            int_ = cntx.get_intent([], trust_mode=True, verb=False)
            ext_ = cntx.get_extent(int_, trust_mode=True, verb=False)
            concepts.add(Concept(ext_, int_))

        return concepts

    def _construct_lattice_connections(self, use_tqdm=True):
        n_concepts = len(self._concepts)
        cncpts_map = {c._idx: c for c in self._concepts}
        all_low_neighbs = {c._idx: set() for c in self._concepts}

        for cncpt_idx in tqdm(range(n_concepts - 1, -1, -1), disable=not use_tqdm):
            concept = cncpts_map[cncpt_idx]
            concept._low_neighbs = set()
            possible_neighbs = set(range(cncpt_idx + 1, n_concepts))

            while len(possible_neighbs) > 0:
                pn_idx = min(possible_neighbs)
                possible_neighbs.remove(pn_idx)

                if cncpts_map[pn_idx].is_subconcept_of(concept):
                    all_low_neighbs[cncpt_idx] = all_low_neighbs[cncpt_idx] | {pn_idx} | all_low_neighbs[pn_idx]
                    concept._low_neighbs.add(pn_idx)
                    possible_neighbs = possible_neighbs - all_low_neighbs[pn_idx]

            concept._up_neighbs = set()
            for ln_idx in concept._low_neighbs:
                cncpts_map[ln_idx]._up_neighbs.add(concept._idx)

    def _find_new_concept_objatr(self):
        cncpt_dict = {c._idx: c for c in self._concepts}
        for c in self._concepts:
            c._new_attrs = tuple(
                set(c.get_intent()) - {m for un_idx in c.get_upper_neighbs() for m in cncpt_dict[un_idx].get_intent()})
            c._new_objs = tuple(
                set(c.get_extent()) - {m for ln_idx in c.get_lower_neighbs() for m in cncpt_dict[ln_idx].get_extent()})

    def _calc_concept_levels(self):
        concepts = self.sort_concepts(self._concepts)

        self._top_concept = concepts[0]
        # concepts_to_check = [self._top_concept]
        concepts[0]._level = 0
        for c in concepts[1:]:
            c._level = max([concepts[un]._level for un in c._up_neighbs]) + 1

    def construct_lattice(self, use_tqdm=False):
        self._construct_lattice_connections(use_tqdm)
        self._calc_concept_levels()
        self._find_new_concept_objatr()

    def _get_metric(self, c, m):
        if c._metrics is not None:
            for k, v in c._metrics.items():
                if k == m:
                    return v
        return None

    def _get_concepts_position(self, concepts, level_widths, level_sort, sort_by):
        pos = {}
        max_width = max(level_widths.values())
        n_levels = len(level_widths)

        last_level = None
        cur_level_idx = None
        for c in sorted(concepts, key=lambda c: c.get_level()):
            cl = c.get_level()
            cur_level_idx = cur_level_idx + 1 if cl == last_level else 1
            last_level = cl
            pos[c.get_id()] = (cur_level_idx - level_widths[cl] / 2 - 0.5, n_levels - cl)

        if level_sort is not None and sort_by is not None:
            level_sort = n_levels // 2 if level_sort == 'mean' else level_sort

            cncpt_by_levels = {}
            for c in concepts:
                cncpt_by_levels[c.get_level()] = cncpt_by_levels.get(c.get_level(), []) + [c]
            pos = {}

            # raise ValueError(f'Unknown feature to sort by: {sort_by}')

            if level_sort == 'all':
                for cl in range(0, len(level_widths)):
                    for c_l_idx, c in enumerate(sorted(cncpt_by_levels[cl], key=lambda c: self._get_metric(c, sort_by))):
                        pos[c.get_id()] = (c_l_idx - level_widths[cl] / 2 + 0.5, n_levels - cl)
            else:
                cl = level_sort
                for c_l_idx, c in enumerate(
                        sorted(cncpt_by_levels[level_sort], key=lambda c: self._get_metric(c, sort_by))):
                    pos[c.get_id()] = (c_l_idx - level_widths[cl] / 2 + 0.5, n_levels - cl)

                for cl in range(level_sort - 1, -1, -1):
                    for c_l_idx, c in enumerate(cncpt_by_levels[cl]):
                        pos[c.get_id()] = (np.mean([pos[ln][0] for ln in c.get_lower_neighbs() if ln in pos]), n_levels - cl)

                for cl in range(level_sort + 1, n_levels):
                    for c_l_idx, c in enumerate(cncpt_by_levels[cl]):
                        pos[c.get_id()] = (np.mean([pos[un][0] for un in c.get_upper_neighbs() if un in pos]), n_levels - cl)

            # center to 0
            for cl in range(n_levels):
                m = np.mean([pos[c._idx][0] for c in cncpt_by_levels[cl]])
                for c_l_idx, c in enumerate(sorted(cncpt_by_levels[cl], key=lambda c: pos[c._idx][0])):
                    pos[c._idx] = (pos[c._idx][0] - m, pos[c._idx][1])
                    pos[c._idx] = (c_l_idx - level_widths[cl] / 2 + 0.5, n_levels - cl)

        return pos

    def get_plotly_fig(self, level_sort=None, sort_by=None, y_precision=None, color_by=None, title=None,
                       cbar_title=None, cmin=None, cmid=None, cmax=None, cmap='RdBu',
                       new_attrs_lim=5, new_objs_lim=5,
                       metrics_to_print='all', figsize=None):
        connections_dict = {}
        for c in self.get_concepts():
            connections_dict[c.get_id()] = [ln_idx for ln_idx in c.get_lower_neighbs()]
        level_widths = {}
        for c in self.get_concepts():
            level_widths[c.get_level()] = level_widths.get(c.get_level(), 0) + 1

        pos = self._get_concepts_position(self.get_concepts(), level_widths, level_sort, sort_by)

        G = nx.from_dict_of_lists(connections_dict)
        nx.set_node_attributes(G, pos, 'pos')

        edge_x = [y for edge in G.edges() for y in [pos[edge[0]][0], pos[edge[1]][0], None]]
        edge_y = [y for edge in G.edges() for y in [pos[edge[0]][1], pos[edge[1]][1], None]]

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            textposition='middle right',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbw' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale=cmap,
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title=cbar_title if cbar_title else 'y_mean',
                    xanchor='left',
                    titleside='right',
                ),
                line_width=2))
        node_adjacencies = []
        node_text = []
        node_color = []
        node_title = []
        # for node, adjacencies in enumerate(G.adjacency()):
        for node in G.nodes():
            c = self.get_concept_by_id(node)
            node_color.append(get_not_none(self._get_metric(c, color_by), 'grey'))
            # node_color.append(c._mean_y if c._mean_y is not None else 'grey')
            # node_text.append('a\nbc')
            node_text.append(c.pretty_repr(metrics_to_print=metrics_to_print).replace('\n', '<br>') + \
                             '')  # f'\npos({pos[c._idx]})')
            new_attrs_str = '' if len(c._new_attrs) == 0 else \
                f"{','.join(c._new_attrs) if c._new_attrs else ''}" if len(c._new_attrs) < new_attrs_lim \
                    else f"n: {len(c._new_attrs)}"
            new_objs_str = '' if len(c._new_objs) == 0 else \
                f"{','.join(c._new_objs) if c._new_objs else ''}" if len(c._new_objs) < new_objs_lim \
                    else f"n: {len(c._new_objs)}"
            node_title.append(new_attrs_str + '<br>' + new_objs_str)
        # node_text.append('# of connections: '+str(len(adjacencies[1])))

        node_trace.marker.color = node_color
        node_trace.hovertext = node_text
        node_trace.text = node_title

        if cmid is not None:
            node_trace.marker.cmid = cmid

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=title if title else 'Concept Lattice',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            # annotations=[ dict(
                            #    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                            #    showarrow=False,
                            #    xref="paper", yref="paper",
                            #    x=0.005, y=-0.002 ) ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=figsize[0] if figsize is not None else 1000,
                            height=figsize[1] if figsize is not None else 500,
                        )
                        )
        return fig