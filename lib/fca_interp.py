import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import concepts as concepts_mit
import networkx as nx
import plotly.graph_objects as go

from itertools import combinations, chain

from formal_context import Concept, BinaryContext, Binarizer
from pattern_structure import MultiValuedContext

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
                           is_monotonic=False):
        if algo == 'CBO':
            concepts = self._close_by_one(max_iters_num, max_num_attrs, min_num_objs, use_tqdm,
                                          is_monotonic=is_monotonic)
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
                f"Given task type {self._task_type} is not supported. Possible values are 'regression',"
                f"'binary classification'")
        ms['y_pred_mean'] = np.mean(y_pred)
        ms['y_true_mean'] = np.mean(y_true)
        return ms

    def _close_by_one(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm, is_monotonic=False):
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

            new_combs = [int_ + [x] for x in range((comb[-1] if len(comb) > 0 else -1) + 1, n_attrs) if x not in int_]
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
        cncpts_map = {c.get_id(): c for c in self._concepts}
        all_low_neighbs = {c.get_id(): set() for c in self._concepts}

        for cncpt_idx in tqdm(sorted(cncpts_map.keys(), key=lambda idx: -idx), disable=not use_tqdm):
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

    def _find_new_concept_objatr(self):
        cncpt_dict = {c._idx: c for c in self._concepts}
        for c in self._concepts:
            if c.get_upper_neighbs() is not None:
                c._new_attrs = tuple(
                    set(c.get_intent()) - {m for un_idx in c.get_upper_neighbs()
                                           for m in cncpt_dict[un_idx].get_intent()})
            else:
                c._new_attrs = tuple(c.get_intent())
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

    def calc_strongness(self, cntx_full, use_tqdm=False):
        for c in tqdm(self.get_concepts(), disable=not use_tqdm):
            c._metrics['strongness'] = len(c.get_extent())/len(cntx_full.get_extent([str(x) for x in c.get_intent()]))\
                if len(c.get_extent())>0 else 0

    def filter_concepts(self, fltr):
        for c in self._concepts:
            if not fltr(c):
                self.delete_concept(c.get_id())
