import numpy as np
import pandas as pd
import scipy as sp
from tqdm.notebook import tqdm
import concepts as concepts_mit
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime
from frozendict import frozendict
import statistics
import json
from copy import deepcopy, copy
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from joblib import Parallel, delayed

from .formal_context import Concept, BinaryContext, Binarizer
from .pattern_structure import PatternStructure, MultiValuedContext
from .utils_ import get_not_none, sparse_unique_columns



class FormalManager:
    def __init__(self, context, ds_obj=None, target_attr=None, cat_feats=None, task_type=None, context_full=None,
                 n_jobs=1):
        self._context = context
        if ds_obj is not None:
            ds_obj.index = ds_obj.index.astype(str)
        self._ds_obj = ds_obj
        self._concepts = None
        self._target_attr = target_attr
        self._top_concept = None
        self._cat_feats = cat_feats
        self._task_type = task_type
        self._context_full = context_full
        self._n_jobs = n_jobs

    def get_context(self):
        return self._context

    def get_context_full(self):
        return self._context_full

    def get_concepts(self):
        return self._concepts

    def get_concept_by_id(self, id_):
        cpt = [c for c in self._concepts if c.get_id() == id_]
        if len(cpt) == 0:
            return None
        return cpt[0]

    def sort_concepts(self, concepts=None):
        concepts = get_not_none(concepts, self._concepts)
        #return sorted(concepts, key=lambda c: (len(c.get_intent()), ','.join(c.get_intent())))
        #return sorted(concepts,
        ##              key=lambda c: (len(c._get_intent_as_array(c.get_intent())),
         #                            ','.join([str(m) for m in c._get_intent_as_array(c.get_intent())]))
         #             )
        return sorted(concepts,
                  key=lambda c: (-len(c.get_extent()),
                                ','.join([str(g) for g in c.get_extent()]))
                 )

    def construct_concepts(self, algo='mit', max_iters_num=None, max_num_attrs=None, min_num_objs=None, use_tqdm=True,
                           is_monotonic=False, strongness_lower_bound=None, calc_metrics=True,
                           strong_concepts=None, n_bootstrap_epochs=500, sample_size_bootstrap=15,
                           n_best_bootstrap_concepts=None, agglomerative_strongness_delta=0.1,
                           stab_min_bound_bootstrap=None, y_type='True', rf_params={}, rf_class=None):
        if algo == 'RandomForest':
            concepts = self._random_forest_concepts(y_type=y_type, rf_params=rf_params, rf_class=rf_class)
            concepts = set(concepts)
        elif algo == 'FromMaxConcepts_Bootstrap':
            max_cncpts = self.construct_max_strong_hyps(verb=True)
            concepts, min_concepts_by_iters = self._agglomerative_concepts_construction(
                max_cncpts,
                strongness_delta=agglomerative_strongness_delta,
                stab_min_bound=stab_min_bound_bootstrap,
                bootstrap_sample_size=sample_size_bootstrap,
                use_tqdm=use_tqdm,
                n_epochs_bootstrap=n_bootstrap_epochs,
                n_best_concepts_bootstrap=n_best_bootstrap_concepts,
                verb=False)
            self.min_concepts_by_iters = min_concepts_by_iters
        elif algo == 'Agglomerative_Bootstrap':
            #concepts = self._unite_bootstrap_concepts(strong_concepts, n_epochs=n_bootstrap_epochs,
            #                                          sample_size=sample_size_bootstrap, use_tqdm=use_tqdm,
            #                                          stab_min_bound=stab_min_bound_bootstrap, verb=False,
            #                                          strongness_min_bound=strongness_lower_bound,
            #                                          n_best_concepts=n_best_bootstrap_concepts, is_monotonic=is_monotonic)
            concepts, min_concepts_by_iters = self._agglomerative_concepts_construction(
                strong_concepts,
                strongness_delta=agglomerative_strongness_delta,
                stab_min_bound=stab_min_bound_bootstrap,
                bootstrap_sample_size=sample_size_bootstrap,
                use_tqdm=use_tqdm,
                n_epochs_bootstrap=n_bootstrap_epochs,
                n_best_concepts_bootstrap=n_best_bootstrap_concepts,
                verb=False)
            self.min_concepts_by_iters = min_concepts_by_iters
        elif algo == 'FromMaxConcepts':
            max_cncpts = self.construct_max_strong_hyps(verb=True)
            concepts = self._close_by_one_concepts(max_cncpts, is_monotonic,
                                                   strongness_min_bound=strongness_lower_bound, verb=False)
            concepts = set(concepts)
        elif algo == 'Agglomerative':
            concepts = self._close_by_one_concepts(strong_concepts, is_monotonic,
                                                   strongness_min_bound=strongness_lower_bound, verb=False)
            concepts = set(concepts)
        elif isinstance(self._context, MultiValuedContext):
            concepts = self._close_by_one_pattern_structure(max_iters_num, max_num_attrs, min_num_objs, use_tqdm,
                                          is_monotonic=is_monotonic)
            #concepts = {c for c in concepts}
            concepts = {PatternStructure( tuple(self._context.get_objs()[c.get_extent()])
                                        if len(c.get_extent()) > 0 else tuple(),
                                         {self._context.get_attrs()[k] if type(k) not in [str, np.str_] else k: v for k, v in c.get_intent().items() }
                                         if c.get_intent() is not None else tuple()
                                        ) for c in concepts}
        elif strongness_lower_bound is not None:
            concepts = self._close_by_one_strong_limit(max_iters_num, max_num_attrs, min_num_objs, use_tqdm,
                                          is_monotonic=is_monotonic, strongness_lower_bound=strongness_lower_bound)
            concepts = {Concept(tuple(self._context.get_objs(is_full=False)[c.get_extent()])
                                if len(c.get_extent()) > 0 else tuple(),
                                tuple(self._context.get_attrs()[c.get_intent()])
                                if len(c.get_intent()) > 0 else tuple()
                                ) for c in concepts}
            concepts = set(concepts)
        elif algo == 'CBO':
            concepts = self._close_by_one(max_iters_num, max_num_attrs, min_num_objs, use_tqdm,
                                          is_monotonic=is_monotonic,)
            concepts = {Concept(tuple(self._context.get_objs(is_full=False)[c.get_extent()])
                                   if len(c.get_extent()) > 0 else tuple(),
                                tuple(self._context.get_attrs()[c.get_intent()])
                                    if len(c.get_intent()) > 0 else tuple()
                                ) for c in concepts}
            concepts = set(concepts)
        elif algo == 'mit':
            concepts = self._concepts_by_mit()
            #d_objs = {g: idx for idx, g in enumerate(self._context.get_objs(is_full=False))}
            #d_attrs = {m: idx for idx, m in enumerate(self._context.get_attrs(is_full=False))}
            #concepts = {Concept(tuple([d_objs[g] for g in c.get_extent()] if len(c.get_extent())>0 else []),
            #                    tuple([d_attrs[m] for m in c.get_intent()] if len(c.get_intent())>0 else [],)
            #                    ) for c in concepts}
        else:
            raise ValueError('The only supported algorithm is CBO (CloseByOne) and "mit" (from library "concepts")')

        if is_monotonic:
            concepts = {Concept([g for g in self._context.get_objs(is_full=False) if g not in c.get_extent()], c.get_intent())
                        for c in concepts}

        new_concepts = set()
        for idx, c in tqdm(enumerate(self.sort_concepts(concepts)), desc='Postprocessing', disable=not use_tqdm,
                           total=len(concepts)):
            ext_short = c.get_extent()
            int_short = c.get_intent()

            if not isinstance(self._context, MultiValuedContext):
                int_ = self._context.get_intent(ext_short, is_full=True, verb=True)
                ext_ = self._context.get_extent(int_short, is_full=True, verb=True)
                #int_ = [m_ for m in int_short for m_ in [m] + list(self._context.get_same_attrs(m))]
                #ext_ = [g_ for g in ext_short for g_ in [g] + list(self._context.get_same_objs(g))]
                metrics = self._calc_metrics_inconcept(ext_) if len(ext_) > 0 and calc_metrics else None
                new_concepts.add(Concept(ext_, int_, idx=idx,
                                     metrics=metrics,
                                     extent_short=ext_short, intent_short=int_short,
                                     is_monotonic=is_monotonic))
            else:
                #int_ = self._context.get_intent(ext_short, verb=True, trust_mode=False)
                #ext_ = self._context.get_extent(int_short, verb=True, trust_mode=False)
                ext_ = ext_short
                int_ = int_short
                metrics = self._calc_metrics_inconcept(ext_) if len(ext_) > 0 and calc_metrics else None
                c._metrics = metrics
                c._idx = idx
                #new_concepts.add(PatternStructure(ext_, int_, idx=idx,
                #                    metrics=metrics,
                #                    is_monotonic=is_monotonic,
                #                    cat_feats=self._cat_feats))
        #concepts = new_concepts
        self._concepts = concepts

        self._top_concept = self.get_concept_by_id(0)

    def delete_concept(self, c_idx):
        c = self.get_concept_by_id(c_idx)
        upns, lns = c.get_upper_neighbs(), c.get_lower_neighbs()
        #if upns is None or len(upns) == 0:
        #    raise KeyError(f'Cannot delete concept {c_idx}. It may be supremum')

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

        ms = {}

        y_true = self._context.get_y_true(ext_)
        if y_true is not None:
            ms['mean_y_true'] = np.mean(y_true)

        y_pred = self._context.get_y_pred(ext_)
        if y_pred is not None:
            ms['mean_y_pred'] = np.mean(y_pred)

        if y_true is None or y_pred is None:
            return ms
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
                f"Given task type {self._task_type} is not supported. Possible values are 'regression',"
                f"'binary classification'")
        return ms

    def _close_by_one(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm, is_monotonic=False,
                      iter_by_objects=False, ):
        cntx = self._context
        n_attrs = len(cntx.get_attrs(is_full=False))

        combs_to_check = [[]]
        concepts = set()
        if use_tqdm:
            if max_num_attrs is not None and min_num_objs is not None:
                tot = min(sum([sp.special.comb(N=n_attrs, k=x) for x in range(0, max_num_attrs + 1)]),
                          sum([sp.special.comb(N=n_attrs, k=x) for x in range(0, min_num_objs + 1)]))
                t = tqdm(total=tot)
            elif max_num_attrs is not None:
                t = tqdm(total=sum([sp.special.comb(N=n_attrs, k=x) for x in range(0, max_num_attrs + 1)]))
            elif min_num_objs is not None:
                t = tqdm(total=sum([sp.special.comb(N=n_attrs, k=x) for x in range(0, min_num_objs + 1)]))
            else:
                t = tqdm(total=len(combs_to_check))
            t.set_description('Calculating concepts')

        iter = 0
        saved_ints = set()
        while len(combs_to_check) > 0:
            iter += 1
            if max_iters_num is not None and iter >= max_iters_num:
                break
            comb = combs_to_check.pop(0)

            ext_ = cntx.get_extent(comb, trust_mode=True, verb=False)
            int_ = cntx.get_intent(ext_, trust_mode=True, verb=False)
            #print(comb)
            #print(ext_)
            #print(int_)
            #print('====================')

            if max_num_attrs is not None and len(comb) > max_num_attrs:
                continue
            if min_num_objs is not None and len(ext_) <= min_num_objs:
                continue

            new_int_ = [x for x in int_ if x not in comb]

            if (len(comb) > 0 and any([x < comb[-1] for x in new_int_])) or tuple(int_) in saved_ints:
                if use_tqdm:
                    t.update()
                continue

            concepts.add(Concept(ext_, int_))
            saved_ints.add(tuple(int_))

            new_combs = [int_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_attrs) if x not in int_]
            combs_to_check = new_combs + combs_to_check
            if use_tqdm:
                t.update()
                if max_num_attrs is None or min_num_objs is not None:
                    t.total += len(new_combs)

        if use_tqdm:
            t.close()

        if tuple([]) not in saved_ints:
            int_ = cntx.get_intent([], trust_mode=True, verb=False)
            ext_ = cntx.get_extent(int_, trust_mode=True, verb=False)
            concepts.add(Concept(ext_, int_))

        return concepts

    def _close_by_one_strong_limit(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm, is_monotonic=False,
                      iter_by_objects=False, strongness_lower_bound=None):
        cntx = self._context
        n_objs = len(cntx.get_objs(is_full=False))

        combs_to_check = [[]]
        concepts = set()
        if use_tqdm:
            if max_num_attrs is not None and min_num_objs is not None:
                tot = min(sum([sp.misc.comb(N=n_objs, k=x) for x in range(0, max_num_attrs + 1)]),
                          sum([sp.misc.comb(N=n_objs, k=x) for x in range(0, min_num_objs + 1)]))
                t = tqdm(total=tot)
            elif max_num_attrs is not None:
                t = tqdm(total=sum([sp.misc.comb(N=n_objs, k=x) for x in range(0, max_num_attrs + 1)]))
            elif min_num_objs is not None:
                t = tqdm(total=sum([sp.misc.comb(N=n_objs, k=x) for x in range(0, min_num_objs + 1)]))
            else:
                t = tqdm(total=len(combs_to_check))

        iter = 0
        saved_exts = set()
        while len(combs_to_check) > 0:
            iter += 1
            if max_iters_num is not None and iter >= max_iters_num:
                break
            comb = combs_to_check.pop(0)

            int_ = cntx.get_intent(comb, trust_mode=True, verb=False)
            ext_ = cntx.get_extent(int_, trust_mode=True, verb=False)
            print(comb)
            print(ext_)
            print(int_)
            print('====================')

            if max_num_attrs is not None and len(comb) > max_num_attrs:
                continue
            if min_num_objs is not None and len(ext_) <= min_num_objs:
                continue

            new_ext_ = [x for x in ext_ if x not in comb]

            if (len(comb) > 0 and any([x < comb[-1] for x in new_ext_])) or tuple(ext_) in saved_exts:
                if use_tqdm:
                    t.update()
                continue

            if strongness_lower_bound is not None and len(ext_)>0:
                ext_verb = cntx.get_extent(int_, trust_mode=True, verb=True,)
                int_verb = cntx.get_intent(ext_verb, is_full=True)

                ext_full = cntx.get_extent(int_verb,is_full=True)
                ext_full_full = self._context_full.get_extent(int_verb, is_full=True)
                strongness = len(ext_full)/len(ext_full_full) if len(ext_full) > 0 else 0
                if strongness < strongness_lower_bound:
                    continue

            concepts.add(Concept(ext_, int_))
            saved_exts.add(tuple(ext_))

            new_combs = [ext_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_objs) if x not in ext_]
            combs_to_check = new_combs + combs_to_check
            if use_tqdm:
                t.update()
                if max_num_attrs is None or min_num_objs is not None:
                    t.total += len(new_combs)

#        if tuple([]) not in saved_exts:
#            int_ = cntx.get_intent([], trust_mode=True, verb=False)
#            ext_ = cntx.get_extent(int_, trust_mode=True, verb=False)
#            concepts.add(Concept(ext_, int_))

        return concepts

    def _close_by_one_pattern_structure(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm, is_monotonic=False):
        cntx = self._context
        n_objs = len(cntx.get_objs())

        combs_to_check = [[g_idx] for g_idx in range(len(cntx._objs))]
        concepts = set()
        iter_ = 0
        saved_exts = set()
        # print(combs_to_check)
        #t0 = datetime.now()
        while len(combs_to_check) > 0:
            iter_ += 1

            comb = combs_to_check.pop(0)

            int_ = cntx.get_intent(comb, trust_mode=True, verb=False)
            ext_ = cntx.get_extent(int_, trust_mode=True, verb=False)
            new_ext_ = [x for x in ext_ if x not in comb]

            #t1 = datetime.now()
            #dt = (t1 - t0).total_seconds()
            #print(f"{iter_}: len(ext_)={len(ext_)}, first_in_comb: {ext_[0]}, new_ext_len:{len(new_ext_)}, time spend: {dt:.2f} sec, speed: {dt / iter_:.2f} sec/iter")

            if (len(comb) > 0 and any([x < comb[-1] for x in new_ext_])) or tuple(ext_) in saved_exts:
                continue

            concepts.add(PatternStructure(ext_, int_, cat_feats=self._cat_feats))
            saved_exts.add(tuple(ext_))

            new_combs = [ext_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_objs) if x not in ext_]
            combs_to_check = new_combs + combs_to_check
        return concepts

    def _close_by_one_concepts(self, strong_concepts, is_monotonic=False, strongness_min_bound=0.5,
                               verb=True):
        cntx = self._context
        cntx_full = self._context_full

        concept_class = Concept if isinstance(cntx, BinaryContext) \
            else PatternStructure if isinstance(cntx, MultiValuedContext) \
            else None
        assert concept_class is not None, 'Context class is not recognized'


        n_concepts = len(strong_concepts)

        combs_to_check = [[g_idx] for g_idx in range(len(strong_concepts))]
        concepts = set()
        iter_ = 0
        saved_combs = set()

        t0 = datetime.now()
        while len(combs_to_check) > 0:
            iter_ += 1

            comb = combs_to_check.pop(0)

            try:
                ext_united = list(set([g for c_idx in comb for g in strong_concepts[
                    c_idx].get_extent()]))  # transforming concepts to their common extents
            except Exception as e:
                if verb:
                    print(comb, e)
                raise e
            if concept_class == Concept:
                int_ = cntx.get_intent(ext_united, trust_mode=False, verb=True, is_full=True)
                ext_ = cntx.get_extent(int_, trust_mode=False, verb=True, is_full=True)
            else:
                int_ = cntx.get_intent(ext_united, trust_mode=False, verb=True, )
                ext_ = cntx.get_extent(int_, trust_mode=False, verb=True)

            comb_ = [idx for idx, concept in enumerate(strong_concepts) if
                     all([g in ext_ for g in concept.get_extent()])]  # transforming extents to their concepts

            new_comb_ = [x for x in comb_ if x not in comb]

            t1 = datetime.now()
            dt = (t1 - t0).total_seconds()
            try:
                if verb:
                    print(
                        f"{iter_}: len(comb_)={len(comb_)}, first_in_comb: {comb_[0]}, new_comb_len:{len(new_comb_)}, time spend: {dt:.2f} sec, speed: {dt / iter_:.2f} sec/iter")
            except Exception as e:
                if verb:
                    print(comb, comb_, e)
                raise e

            if (len(comb) > 0 and any([x < comb[-1] for x in new_comb_])) or tuple(comb_) in saved_combs:
                continue

            if strongness_min_bound is not None:
                if concept_class == Concept:
                    ext_full = cntx.get_extent(int_, trust_mode=False, verb=True, is_full=True)
                    ext_full_ = cntx_full.get_extent(int_, trust_mode=False, verb=True, is_full=True)
                else:
                    ext_full = cntx.get_extent(int_, trust_mode=False, verb=True)
                    ext_full_ = cntx_full.get_extent(int_, trust_mode=False, verb=True)
                strongness = len(ext_full) / len(ext_full_) if len(ext_full) > 0 else 0
                if strongness < strongness_min_bound:
                    continue

            if concept_class == Concept:
                c = concept_class(ext_, int_)
            else:
                c = concept_class(ext_, int_, cat_feats=cntx._attrs[cntx._cat_attrs_idxs])
            concepts.add(c)

            saved_combs.add(tuple(comb_))

            new_combs = [comb_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_concepts) if
                         x not in comb_]
            combs_to_check = new_combs + combs_to_check

        return concepts

    def _construct_lattice_connections(self, use_tqdm=True):
        n_concepts = len(self._concepts)
        cncpts_map = {c.get_id(): c for c in self._concepts}
        all_low_neighbs = {c.get_id(): set() for c in self._concepts}

        for cncpt_idx in tqdm(sorted(cncpts_map.keys(), key=lambda idx: -idx), disable=not use_tqdm, desc='construct lattice connections'):
            concept = cncpts_map[cncpt_idx]
            concept._low_neighbs = set()
            possible_neighbs = set([idx for idx in cncpts_map.keys() if idx>cncpt_idx])

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

    def _construct_spanning_tree(self, use_tqdm=True):
        n_concepts = len(self._concepts)
        cncpts_map = {c.get_id(): c for c in self._concepts}

        for cncpt_idx in tqdm(sorted(cncpts_map.keys(), key=lambda idx: idx),
                              disable=not use_tqdm, desc='construct spanning tree'):
            c = cncpts_map[cncpt_idx]
            c._low_neighbs_st = set()
            if cncpt_idx == 0:
                c._up_neighb_st = None
                continue

            un_idx = 0
            sifted = True
            while sifted:
                un = cncpts_map[un_idx]
                for ln_idx in un._low_neighbs_st:
                    ln = cncpts_map[ln_idx]
                    if c.is_subconcept_of(ln):
                        un_idx = ln_idx
                        sifted = True
                        break
                else:
                    sifted = False
            un._low_neighbs_st.add(cncpt_idx)
            c._up_neighb_st = un_idx


            #if c._up_neighbs is not None and len(c._up_neighbs)>0:
            #    pn_idx = min(c._up_neighbs)
            #    c._up_neighb_st = pn_idx
            #    cncpts_map[pn_idx]._low_neighbs_st.add(cncpt_idx)
            #    continue

            #pn_idx = cncpt_idx-1
            #while pn_idx >= 0:
            #    if c.is_subconcept_of(cncpts_map[pn_idx]):
            #        c._up_neighb_st = pn_idx
            #        cncpts_map[pn_idx]._low_neighbs_st.add(cncpt_idx)

            #        break

            #    pn_idx -= 1
            #else:
            #    c._up_neighb_st = None

    def _construct_spanning_tree_old(self, use_tqdm=True):
        n_concepts = len(self._concepts)
        cncpts_map = {c.get_id(): c for c in self._concepts}

        for cncpt_idx in tqdm(sorted(cncpts_map.keys(), key=lambda idx: idx),
                              disable=not use_tqdm, desc='construct spanning tree'):
            c = cncpts_map[cncpt_idx]
            c._low_neighbs_st = set()
            if cncpt_idx == 0:
                c._up_neighb_st = None
                continue

            if c._up_neighbs is not None and len(c._up_neighbs)>0:
                pn_idx = min(c._up_neighbs)
                c._up_neighb_st = pn_idx
                cncpts_map[pn_idx]._low_neighbs_st.add(cncpt_idx)
                continue

            pn_idx = cncpt_idx-1
            while pn_idx >= 0:
                if c.is_subconcept_of(cncpts_map[pn_idx]):
                    c._up_neighb_st = pn_idx
                    cncpts_map[pn_idx]._low_neighbs_st.add(cncpt_idx)

                    break

                pn_idx -= 1
            else:
                c._up_neighb_st = None


    def _random_forest_concepts(self,  y_type='True', rf_params={}, rf_class=None):
        cntx = self._context
        #X = cntx.get_data().copy()
        X = cntx._data.copy()
        if type(cntx) == MultiValuedContext:
            for f_idx in cntx._cat_attrs_idxs:
                le = LabelEncoder()
                X[:, f_idx] = le.fit_transform(X[:, f_idx])

        y = cntx._y_true if y_type == 'True' else cntx._y_pred

        if rf_class is None:
            if len(set(y)) == 2:
                rf_class = RandomForestClassifier
            else:
                rf_class = RandomForestRegressor

        rf = rf_class(**rf_params)
        rf.fit(X, y)
        preds_rf = rf.predict(X)
        #ds['preds_rf'] = rf.predict(X)
        exts = self._parse_tree_to_extents(rf, X, cntx._objs_full, self._n_jobs)

        concepts = []
        for ext in exts:
            concept_class = Concept if type(cntx) == BinaryContext else PatternStructure
            c = concept_class(ext, cntx.get_intent(ext))
            concepts.append(c)
        return concepts

    def _random_forest_concepts_old(self,  y_type='True', rf_params={}, rf_class=None):
        cntx = self._context
        #X = cntx.get_data().copy()
        X = cntx._data.copy()
        if type(cntx) == MultiValuedContext:
            for f_idx in cntx._cat_attrs_idxs:
                le = LabelEncoder()
                X[:, f_idx] = le.fit_transform(X[:, f_idx])

        y = cntx._y_true if y_type == 'True' else cntx._y_pred

        if rf_class is None:
            if len(set(y)) == 2:
                rf_class = RandomForestClassifier
            else:
                rf_class = RandomForestRegressor

        rf = rf_class(**rf_params)
        rf.fit(X, y)
        preds_rf = rf.predict(X)
        #ds['preds_rf'] = rf.predict(X)
        exts_by_estim = Parallel(-1)(delayed(self._parse_tree_to_extents_old)(est, X, cntx._objs_full) for est in rf.estimators_)
        exts = set()
        for exts_be in exts_by_estim:
            exts |= set(exts_be)
        exts = list(exts)
        #exts = list(set([
        #    tuple(sorted(ext)) for est in rf.estimators_
        #    for ext in self._parse_tree_to_extents(est, X, cntx._objs_full)]
        #))
        #exts = list(set([tuple(cntx.get_extent(cntx.get_intent(ext))) for ext in exts]))

        concepts = []
        for ext in exts:
            concept_class = Concept if type(cntx) == BinaryContext else PatternStructure
            c = concept_class(ext, cntx.get_intent(ext))
            concepts.append(c)
        return concepts



    @staticmethod
    def _parse_tree_to_extents(tree, X, objs, n_jobs=-1):
        if isinstance(tree, (RandomForestClassifier, RandomForestRegressor)):
            paths = tree.decision_path(X)[0].tocsc()
        else:
            paths = tree.decision_path(X).tocsc()

        paths = sparse_unique_columns(paths)[0]
        #f = lambda i, paths: tuple(objs[(paths[:, i] == 1).todense().flatten().tolist()[0]])
        #f = lambda i, paths: tuple(objs[paths.indices[paths.indptr[i]:paths.indptr[i+1]]])
        f = lambda i, paths: paths.indices[paths.indptr[i]:paths.indptr[i + 1]]

        if n_jobs == 1:
            exts = [f(i, paths) for i in range(paths.shape[1])]
        else:
            exts = Parallel(n_jobs)([delayed(f)(i, paths) for i in range(paths.shape[1])])
        return exts

    @staticmethod
    def _parse_tree_to_extents_old(tree, X, objs, n_jobs=-1):
        paths = tree.decision_path(X).tocsc()
        f = lambda i, paths: tuple(objs[(paths[:, i] == 1).todense().flatten().tolist()[0]])

        exts = Parallel(n_jobs)([delayed(f)(i, paths) for i in range(paths.shape[1])])
        return exts

    def _find_new_concept_objatr(self):
        cncpt_dict = {c._idx: c for c in self._concepts}
        for c in self._concepts:

            if c.get_upper_neighbs() is not None and len(c.get_upper_neighbs())>0 and c.get_intent() is not None:
                if type(c) == Concept:
                    c._new_attrs = tuple(
                        set(c.get_intent()) - {m for un_idx in c.get_upper_neighbs()
                                               for m in cncpt_dict[un_idx].get_intent()})
                else:
                    new_attrs = set()
                    try:
                        for k, v in c.get_intent().items():
                            if not any([ cncpt_dict[un_idx].get_intent() is not None
                                and type(cncpt_dict[un_idx].get_intent().get(k, None)) == type(v)
                                and cncpt_dict[un_idx].get_intent().get(k, None) == v
                                         for un_idx in  c.get_upper_neighbs()]):
                                new_attrs.add(k)
                    except Exception as e:
                        print(f'Weird ps {c.get_id()}, objects: {c.get_extent()}, upper_neighbs: {c.get_upper_neighbs()}')
                        raise(e)

                    c._new_attrs = tuple(new_attrs)
            else:
                c._new_attrs = tuple(c.get_intent()) if c.get_intent() is not None else tuple([])

            if c.get_lower_neighbs() is not None:
                c._new_objs = tuple(
                    set(c.get_extent()) - {m for ln_idx in c.get_lower_neighbs()
                                           for m in cncpt_dict[ln_idx].get_extent()})
            else:
                c._new_objs = tuple(c.get_extent())

    def _calc_concept_levels(self):
        concepts = self.sort_concepts(self._concepts)

        self._top_concept = concepts[0]
        # concepts_to_check = [self._top_concept]
        for c in concepts:
            if c.get_upper_neighbs() is None or len(c.get_upper_neighbs())==0:
                c._level = 0
            else:
                c._level = max([self.get_concept_by_id(un_id)._level for un_id in c.get_upper_neighbs()]) + 1

    def construct_lattice(self, use_tqdm=False, only_spanning_tree=False):
        if not only_spanning_tree:
            self._construct_lattice_connections(use_tqdm)
        self._construct_spanning_tree(use_tqdm)
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
            txt_ = c.pretty_repr(metrics_to_print=metrics_to_print).replace('\n', '<br>') + ''
            txt_ = '<br>'.join([x[:50]+('...' if len(x)>50 else '') for x in txt_.split('<br>')])
            node_text.append(txt_)
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

    def calc_stab_bounds(self, cidx,):
        c = self.get_concept_by_id(cidx)
        lns = c.get_lower_neighbs()
        difs = np.array(
            sorted([len(set(c.get_extent()) - set(self.get_concept_by_id(didx).get_extent())) for didx in lns])
            ).astype(float)
        minb = 1 - sum(1 / (2 ** difs))
        maxb = 1 - 1 / (2 ** difs[0])
        return minb, maxb, difs[0]

    def calc_stability_approx(self, use_tqdm=False):
        for c in tqdm(self.get_concepts(), disable=not use_tqdm):
            if len(c.get_lower_neighbs()) > 0:
                minb, maxb, mindif = self.calc_stab_bounds(c.get_id())
            elif len(c.get_extent()) < 0: # TODO: Write function to calculate real stability
                pass
                #print('calc real stability for idx', c.get_id())
                #minb = get_stability(c._idx, fm_bank)
                #maxb = minb
                #mindif = None
            else:
                # raise ValueError(f'Too big extent to calc pure stability ({c._idx} concept)')
                #print((f'Too big extent to calc pure stability ({c._idx} concept)'))
                minb, maxb, mindif = [None] * 3
            c._metrics['stab_min_bound'] = minb
            c._metrics['stab_max_bound'] = maxb
            c._metrics['log_stab_min_bound'] = -np.log2(1 - minb) if minb is not None else None
            c._metrics['log_stab_max_bound'] = -np.log2(1 - maxb) if maxb is not None else None
            c._metrics['lstab_min_bound'] = mindif - len(self.get_context().get_attrs(is_full=True))\
                if mindif is not None else None

    def calc_strongness(self, use_tqdm=False):
        cntx_full = self._context_full
        for c in tqdm(self.get_concepts(), disable=not use_tqdm):
            if type(c) == Concept:
                int_ = [str(x) for x in c.get_intent()]
                assert all([x in cntx_full.get_attrs() for x in int_]), 'Some attributes are missing in cntx_full'
                #int_ = [x for x in int_ if x in cntx_full.get_attrs(is_full=False)]

                ext_ = c.get_extent()
                ext_full = cntx_full.get_extent(int_, is_full=True)
            elif type(c) == PatternStructure:
                ext_ = c.get_extent()
                ext_full = cntx_full.get_extent(c.get_intent())
            c._metrics['strongness'] = len(ext_) / len(ext_full) \
                if len(ext_) > 0 else 0

    def filter_concepts(self, fltr):
        for c in self._concepts:
            if not fltr(c):
                self.delete_concept(c.get_id())

    def calc_cover_of_concepts(self, concepts=None):
        concepts = self.get_concepts() if concepts is None else concepts
        return set([g for c in concepts for g in c.get_extent()])#verb=True)])

    def get_min_concepts(self, concepts=None, use_tqdm=True):
        concepts = self.get_concepts() if concepts is None else concepts

        concepts = sorted(concepts, key=lambda c: -len(c.get_extent()))
        for i in tqdm(range(len(concepts)), disable=not use_tqdm):
            if i >= len(concepts):
                break

            c = concepts[i]
            lns = set([c_ for c_ in concepts if c_ != c and c_.is_subconcept_of(c)])
            concepts = [c_ for c_ in concepts if c_ not in lns]
        return concepts

    def select_smallest_covering_hyps(self, concepts=None, use_tqdm=True, use_pruning=False):
        concepts = self.get_concepts() if concepts is None else concepts

        selected_hyps = []
        n_added = []
        n_covered = []
        for i in tqdm(range(len(concepts)), disable=not use_tqdm):
            cncpts_to_check = [c for c in concepts if c not in selected_hyps]
            if len(cncpts_to_check) == 0:
                break
            cover = self.calc_cover_of_concepts(selected_hyps)
            for c in cncpts_to_check:
                c._metrics['n_uncovered'] = len([g for g in c.get_extent() if g not in cover])
            cncpts_to_check = sorted(cncpts_to_check,
                                     key=lambda c: (-c._metrics['n_uncovered'], -c._metrics['strongness']))
            n_ = cncpts_to_check[0]._metrics['n_uncovered']
            if n_ == 0:
                break
            selected_hyps.append(cncpts_to_check[0])
            n_added.append(n_)
            n_covered.append(len(self.calc_cover_of_concepts(selected_hyps)))

        if use_pruning:
            for i in tqdm(range(len(selected_hyps)), desc='pruning', disable=not use_tqdm):
                cover = len(self.calc_cover_of_concepts(selected_hyps))
                for c in selected_hyps:
                    cover_ = len(self.calc_cover_of_concepts([c_ for c_ in selected_hyps if c_ != c]))
                    if cover_ == cover:
                        selected_hyps = [c_ for c_ in selected_hyps if c_ != c]
                        break
                else:
                    break

        return selected_hyps

    def construct_bin_ds_from_ints(self, ints_, bin_ds_old):
        bin_ds = pd.DataFrame()
        for idx, int_ in enumerate(ints_):
            bin_ds[f'int_{idx}'] = bin_ds_old[int_].all(1)
        bin_ds.index = bin_ds_old.index
        return bin_ds

    def construct_max_strong_hyps(self, use_tqdm=True, verb=True):
        cntx = self._context
        cntx_full = self._context_full

        concept_class = Concept if type(cntx) == BinaryContext else PatternStructure
        objs_to_check = sorted(cntx.get_objs(is_full=True))
        if not verb:
            objs_to_check = cntx._get_ids_in_array(objs_to_check, cntx.get_objs(is_full=True), 'objects')

        concepts = []
        for i in tqdm(range(len(objs_to_check)), disable=not use_tqdm, desc='construct max strong hyps'):
            if len(objs_to_check) == 0:
                break
            g = objs_to_check.pop(0)

            int_ = cntx.get_intent([g], verb=verb, trust_mode=not verb)
            ext_ = cntx.get_extent(int_, verb=verb, trust_mode=not verb)
            ext_full = cntx_full.get_extent(int_, verb=verb, trust_mode=not verb)
            strongness = len(ext_) / len(ext_full) if len(ext_) > 0 else 0
            assert strongness == 1, f'Object {g} is not a strong hypothesis'

            objs_to_check = [g_ for g_ in objs_to_check if g_ not in ext_]
            if concept_class == Concept:
                c = concept_class(ext_, int_, metrics={'strongness': strongness})
            else:
                c = concept_class(ext_, int_, metrics={'strongness': strongness},
                                  cat_feats=list(cntx._attrs[cntx._cat_attrs_idxs]))
            concepts.append(c)
        return concepts

    def _unite_bootstrap_concepts(self, base_concepts, n_epochs=500, sample_size=15, use_tqdm=True,
                                 stab_min_bound=None, verb=False,
                                 strongness_min_bound=0.5, n_best_concepts=None, is_monotonic=False):
        cntx = self._context
        cntx_full = self._context_full

        concepts_bootstrap = []
        for i in tqdm(range(n_epochs), disable=not use_tqdm, desc='boostrap aggregating'):
            np.random.seed(i)
            sample = np.random.choice(base_concepts, size=sample_size, replace=True)
            concepts = self._close_by_one_concepts(sample, is_monotonic=is_monotonic,
                                              strongness_min_bound=strongness_min_bound, verb=verb)
            fm = FormalManager(cntx, context_full=cntx_full)
            fm._concepts = concepts
            fm.calc_strongness()
            for idx, c in enumerate(fm.sort_concepts(concepts)):
                c._idx = idx
            fm.construct_lattice()
            fm.calc_stability_approx()
            if stab_min_bound is not None:
                fm.filter_concepts(lambda c: get_not_none(c._metrics['stab_min_bound'], -100) >= stab_min_bound)

            concepts = list(fm.get_concepts())
            if n_best_concepts is not None:
                concepts = sorted(concepts, key=lambda c:
                    (-get_not_none(c._metrics['stab_min_bound'], -100),
                    -get_not_none(c._metrics['strongness'], -100)))[:n_best_concepts]

            concepts_bootstrap += concepts
        return concepts_bootstrap

    def get_unique_concepts(self, concepts=None):
        concepts = self.get_concepts() if concepts is None else concepts

        saw_ints = set()
        unique_concepts = []
        for c in concepts:
            if c.get_intent() is None:
                continue

            int_ = tuple(c.get_intent()) if type(c) == Concept else frozendict(c.get_intent())
            if int_ not in saw_ints:
                unique_concepts.append(c)
                saw_ints.add(int_)
        return unique_concepts

    def _agglomerative_concepts_construction(self, base_concepts, strongness_delta=0.1, stab_min_bound=0.5,
                                            bootstrap_sample_size=15,
                                            use_tqdm=True, n_epochs_bootstrap=100, n_best_concepts_bootstrap=10,
                                            verb=True):
        cntx = self._context
        cntx_full = self._context_full

        concept_class = Concept if isinstance(cntx, BinaryContext) \
            else PatternStructure if isinstance(cntx, MultiValuedContext) \
            else None
        assert concept_class is not None, 'Context class is not recognized'
        #concept_class = {BinaryContext: Concept, MultiValuedContext: PatternStructure}[type(cntx)]
        if concept_class == Concept:
            int_ = cntx.get_intent([], is_full=True)
            ext_ = cntx.get_extent(int_, is_full=True)
            ext_full = cntx.get_extent(int_, is_full=True)
            bottom_concept = concept_class(ext_, int_,
                                           metrics={'strongness': len(ext_) / len(ext_full) if len(ext_) > 0 else 0})
        else:
            int_ = cntx.get_intent([])
            ext_ = cntx.get_extent(int_)
            ext_full = cntx.get_extent(int_)
            bottom_concept = concept_class(ext_, int_,
                                           metrics={'strongness': len(ext_) / len(ext_full) if len(ext_) > 0 else 0},
                                           cat_feats=cntx._attrs[cntx._cat_attrs_idxs])

        strong_bounds = np.arange(0, 1 + strongness_delta, strongness_delta)[::-1]
        min_concepts_by_iters = {}
        selected_concepts = []

        mc = base_concepts.copy()
        for iter_idx, strong_bound in tqdm(enumerate(strong_bounds), disable=not use_tqdm, total=len(strong_bounds),
                                           desc='agglomerative construction'):
            bs_concepts = self._unite_bootstrap_concepts(mc, sample_size=bootstrap_sample_size,
                                                   stab_min_bound=stab_min_bound,
                                                   n_epochs=n_epochs_bootstrap if n_epochs_bootstrap!='2times' else (len(mc)//bootstrap_sample_size)*2,
                                                   n_best_concepts=n_best_concepts_bootstrap,
                                                   strongness_min_bound=strong_bound)
            unique_concepts = self.get_unique_concepts(bs_concepts)
            if verb:
                print(f'Iter {iter_idx}: num bs unique concepts: {len(unique_concepts)}')

            concepts = unique_concepts + selected_concepts + [bottom_concept] + (base_concepts if iter_idx == 0 else [])
            if verb:
                print(f'Iter {iter_idx}: num concept to fm: {len(concepts)}')

            fm = FormalManager(cntx)
            fm._concepts = concepts
            for idx, c in enumerate(fm.sort_concepts(concepts)):
                c._idx = idx
            fm.construct_lattice(use_tqdm=use_tqdm)
            fm.calc_stability_approx()
            concepts = fm.get_concepts()

            if verb:
                print(f'Iter {iter_idx}: cover of all concepts: {len(self.calc_cover_of_concepts(concepts))}')
            if stab_min_bound is not None:
                stab_concepts = [c for c in concepts if
                                 get_not_none(c._metrics['stab_min_bound'], -100) >= stab_min_bound]
            else:
                stab_concepts = concepts

            if verb:
                print(f'Iter {iter_idx}: cover of stable concepts: {len(self.calc_cover_of_concepts(stab_concepts))}')

            mc = self.select_smallest_covering_hyps(stab_concepts, use_pruning=True)
            if verb:
                print(f'Iter {iter_idx}: cover of min stable concepts: {len(self.calc_cover_of_concepts(mc))}')
                print(f'Iter {iter_idx}: num of min stable concepts: {len(mc)}')
            selected_concepts = self.get_unique_concepts(selected_concepts + mc)
            if verb:
                print(f'Iter {iter_idx}: num of selected concepts: {len(selected_concepts)}')
                print(f'Iter {iter_idx}: cover of selected concepts: {len(self.calc_cover_of_concepts(selected_concepts))}')
            min_concepts_by_iters[iter_idx] = mc
        return selected_concepts, min_concepts_by_iters

    def save_concepts_json(self, fname, concepts=None):
        concepts = self.get_concepts() if concepts is None else concepts

        concepts_json = {}
        for c in concepts:
            c_json = {}
            c_json['extent'] = tuple(c.get_extent())
            c_json['intent'] = tuple(c.get_intent()) if c.get_intent() is not None else None
            c_json['low_neighbs'] = tuple(c.get_lower_neighbs()) if c.get_lower_neighbs() is not None else None
            c_json['up_neighbs'] = tuple(c.get_upper_neighbs()) if c.get_upper_neighbs() is not None else None
            c_json['metrics'] = c._metrics
            concepts_json[c.get_id()] = c_json

        with open(fname, 'w') as f:
            json.dump(concepts_json, f)


    def get_top_concept(self):
        cntx = self._context

        ext_ = list(cntx.get_objs(is_full=True))
        int_ = cntx.get_intent(ext_)
        if type(cntx) == BinaryContext:
            c = Concept(ext_, int_)
        else:
            c = PatternStructure(ext_, int_, cat_feats=cntx._attrs[cntx._cat_attrs_idxs])

        return c

    def get_bottom_concept(self):
        cntx = self._context

        ext_ = []
        int_ = cntx.get_intent(ext_)
        if type(cntx) == BinaryContext:
            c = Concept(ext_, int_)
        else:
            c = PatternStructure(ext_, int_, cat_feats=cntx._attrs[cntx._cat_attrs_idxs])

        return c

    def predict_context(self, cntx, metric='mean_y_true',):
        cncpts_exts = {}
        def get_extent(c):
            if c.get_id() not in cncpts_exts:
                cncpts_exts[c.get_id()] = set(cntx.get_extent(c.get_intent(), verb=False))
            return cncpts_exts[c.get_id()]

        metric = metric if type(metric) == list else [metric]

        cncpts_to_check = set([0])
        cncpts_dict = {c.get_id(): c for c in self.get_concepts()}
        objs_dict = {g: idx for idx, g in enumerate(cntx.get_objs())}
        objs_preds = [[] for g in cntx.get_objs()]
        objs_preds_sum = np.zeros((len(cntx.get_objs()), len(metric)))
        objs_preds_cnt = np.zeros(len(cntx.get_objs()))

        for i in range(len(self.get_concepts())):
            if len(cncpts_to_check) == 0:
                break

            c_id = min(cncpts_to_check)
            cncpts_to_check.remove(c_id)

            c = cncpts_dict[c_id]
            ext = get_extent(c)

            ext_ln = set()
            for ln_id in c._low_neighbs_st: #c.get_lower_neighbs():
                ln = cncpts_dict[ln_id]
                ext_ln |= get_extent(ln)
            ext_to_stop = ext-ext_ln
            preds = [c._metrics[m] for m in metric]
            #objs_preds_sum[[objs_dict[g] for g in ext_to_stop]] += preds
            objs_preds_sum[list(ext_to_stop)] += preds
            objs_preds_cnt += 1
            #for g in ext_to_stop:
            #    g_id = objs_dict[g]
            #    objs_preds[g_id].append(c._metrics[metric])

            cncpts_to_check |= set([ln_id for ln_id in c._low_neighbs_st #c.get_lower_neighbs()
                                    if len(get_extent(cncpts_dict[ln_id]))>0 ] )

        #final_preds = []
        #for preds in objs_preds:
        #    final_preds.append(np.mean(preds))
        final_preds = (objs_preds_sum.T/objs_preds_cnt).T

        return final_preds

    def set_concepts(self, concepts):
        concepts = deepcopy(concepts)
        objs = set(self.get_context().get_objs())
        for c in concepts:
            ext_ = [g for g in c.get_extent() if g in objs]
            int_ = self.get_context().get_intent(ext_)
            ext_ = self.get_context().get_extent(int_)

            c._extent = ext_
            c._intent = int_

        concepts = self.get_unique_concepts(concepts)
        for idx, c in enumerate(self.sort_concepts(concepts)):
            c._idx = idx

        self._concepts = concepts

    def get_metric_difference(self, metric):
        diffs = {}
        for c in self.get_concepts():
            int_ = c.get_intent()
            if int_ is None:
                continue

            un_idxs = c.get_upper_neighbs()
            if un_idxs is None or len(un_idxs) == 0:
                continue
            for un_idx in un_idxs:
                un = self.get_concept_by_id(un_idx)
                un_int = un.get_intent()

                metr_diff = c._metrics[metric] - un._metrics[metric]

                new_int = {}
                for k, v in int_.items():
                    old_v = un_int.get(k)
                    if type(old_v) != type(v) or (old_v != v):
                        new_int[k] = v

                for k in new_int.keys():
                    diffs[k] = diffs.get(k, []) + [metr_diff / len(new_int)]
        return diffs