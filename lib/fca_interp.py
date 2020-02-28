import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import concepts as concepts_mit
import networkx as nx
import plotly.graph_objects as go


class Concept:
	def __init__(self, extent, intent, idx=None, pattern=None, title=None,
				 y_true_mean=None, y_pred_mean=None, metrics=None, extent_short=None, intent_short=None,
				 is_monotonic = False):
		self._extent = np.array(extent)
		self._intent = np.array(intent)
		self._extent_short = np.array(extent_short) if extent_short is not None else self._extent
		self._intent_short = np.array(intent_short) if intent_short is not None else self._intent
		self._low_neighbs = None
		self._up_neighbs = None
		self._idx = idx
		self._title = title
		self._level = None
		self._new_objs = None
		self._new_attrs = None
		self._pattern = pattern
		self._y_true_mean=y_true_mean
		self._y_pred_mean=y_pred_mean
		self._metrics = metrics if metrics is not None else {}
		self._is_monotonic = is_monotonic
		#self._extent_bit = (2**self._extent).sum() if len(self._extent)>0 else 0
		#self._intent_bit = (2**self._intent).sum() if len(self._intent)>0 else 0

	def get_extent(self):
		return self._extent

	def get_intent(self):
		return self._intent

	def __repr__(self):
		s = "Concept"
		if self._idx is not None:
			s += f" {self._idx}"
		if self._title is not None:
			s += f" '{self._title}'"
		s += f"\n"
		s += f"extent (len: {len(self._extent)}): {', '.join([f'{g}' for g in self._extent]) if len(self._extent)>0 else 'emptyset'}\n"
		s += f"intent (len: {len(self._intent)}): {', '.join([f'{m}' for m in self._intent]) if len(self._intent)>0 else 'emptyset'}\n"
		if self._new_objs:
			s += f"new extent (len: {len(self._new_objs)}): {','.join([f'{g}' for g in self._new_objs]) if len(self._new_objs) > 0 else 'emptyset'}\n"
		if self._new_attrs:
			s += f"new intent (len: {len(self._new_attrs)}): {','.join([f'{m}' for m in self._new_attrs]) if len(self._new_attrs) > 0 else 'emptyset'}\n"
		if self._low_neighbs is not None:
			s += f"lower neighbours (len: {len(self._low_neighbs)}): " + \
				 f"{','.join([f'{c}' for c in self._low_neighbs]) if len(self._low_neighbs)>0 else 'emptyset'}\n"
		if self._up_neighbs is not None:
			s += f"upper neighbours (len: {len(self._up_neighbs)}): " + \
				 f"{','.join([f'{c}' for c in self._up_neighbs]) if len(self._up_neighbs)>0 else 'emptyset'}\n"
		s += f"pattern: {self._pattern}\n" if self._pattern is not None else ''
		s += f"level: {self._level}\n" if self._level is not None else ''

		s += f"mean y_true: {self._y_true_mean}\n" if self._y_true_mean is not None else ''
		s += f"mean y_pred: {self._y_pred_mean}\n" if self._y_pred_mean is not None else ''
		s += f"metrics: {self._metrics}\n" if self._metrics is not None else ''
		return s
    
	def pretty_repr(self, print_low_neighbs=False, print_up_neighbs=False, print_level=False, y_precision=None,
					print_mean_y_true=True,print_mean_y_pred=True, metrics_to_print=None):
		s = "Concept"
		if self._idx is not None:
			s += f" {self._idx}"
		if self._title is not None:
			s += f" '{self._title}'"
		s += f"\n"
		def pretty_str(lst, lim=10):
			if len(lst)==0:
				return 'emptyset'
			else:
				return ', '.join([f'{g}' for g in lst[:lim]])+(',...' if len(lst)>lim else '')

		s += f"extent (len: {len(self._extent)}): {pretty_str(self._extent)}\n"
		s += f"intent (len: {len(self._intent)}): {pretty_str(self._intent)}\n"
		if self._new_objs is not None:
			s += f"new extent (len: {len(self._new_objs)}): {pretty_str(self._new_objs)}\n"

		if self._new_attrs is not None:
			s += f"new intent (len: {len(self._new_attrs)}): {pretty_str(self._new_attrs)}\n"

		if print_low_neighbs and self._low_neighbs is not None:
			s += f"lower neighbours (len: {len(self._low_neighbs)}): {pretty_str(self._low_neighbs)}\n"
		if print_up_neighbs and self._up_neighbs is not None:
			s += f"upper neighbours (len: {len(self._up_neighbs)}): {pretty_str(self._up_neighbs)}\n"
		s += f"level: {self._level}\n" if print_level and self._level is not None else ''
		s += f"pattern: {self._pattern}\n" if self._pattern is not None else ''
		if print_mean_y_true and self._y_true_mean is not None:
			s += f"mean_y_true: {np.round(self._y_true_mean,y_precision) if y_precision is not None else self._y_true_mean}\n"
		if print_mean_y_pred and self._y_pred_mean is not None:
			s += f"mean_y_pred: {np.round(self._y_pred_mean, y_precision) if y_precision is not None else self._y_pred_mean}\n"
		if metrics_to_print is not None and self._metrics is not None:
			s += 'Metrics\n'
			for k,v in self._metrics.items():
				if metrics_to_print == 'all' or k in metrics_to_print:
					s+= f'\t{k}: {v}\n'
                    
		pretty_s = ''
		for line in s.split('\n'):
			if len(line)<80:
				pretty_s += line + '\n'
			else:
				pretty_line = ''
				while ' ' in line and len(line)>80:
					if len(pretty_line)>80:
						pretty_s += pretty_line+'\n'
						pretty_line = '\t'
					#print(len(pretty_line),'|', pretty_line)
					#print('---')
					#print(len(line),'|', line)
					#print('======')
					pretty_line += line.split(' ')[0]+' '
					line = ' '.join(line.split(' ')[1:])
				#print(pretty_line)
				#print(line)
				#print('++++++++++++++++++')
				#print('++++++++++++++++++')
				pretty_s += pretty_line
				pretty_s += line+'\n' if len(line)>0 else ''
		return pretty_s

	def __str__(self):
		s = "Concept"
		if self._idx is not None:
			s += f" {self._idx}"
		if self._title is not None:
			s += f" '{self._title}'"
		s += f": ({len(self._extent)} objs, {len(self._intent)} attrs)"
		return s

	def is_subconcept_of(self, c):
		"""if a is subconcept of b, a<=b"""
		if c._is_monotonic:
			return all([g in self._extent_short for g in c._extent_short ])\
				   and all([m in self._intent_short for m in c._intent_short])
		else:
			return all([g in c._extent_short for g in self._extent_short])\
					and all([m in self._intent_short for m in c._intent_short])
		#return self._is_subconcept_of_bit(c)

	def _is_subconcept_of_bit(self, c):
		return (self._extent_bit & c._extent_bit == self._extent_bit)\
			   and (c._intent_bit & self._intent_bit == c._intent_bit)


class Context:
	def __init__(self, data, objs=None, attrs=None, y_true=None, y_pred=None,):# cat_attrs=None):
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
			assert len(y_true) == len(data), f'Data and Y_vals have different num of objects ( Data: {len(data)}, y_vals: {len(y_true)})'
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
		self._data = data[[idx for idx,g in enumerate(objs) if g in same_objs.keys()]][:,
						[idx for idx,m in enumerate(attrs) if m in same_attrs.keys()]]
		self._objs = np.array([g for idx, g in enumerate(objs) if g in same_objs.keys()])
		self._attrs = np.array([m for idx,m in enumerate(attrs) if m in same_attrs.keys()])
		#self._cat_attrs = np.array([m for idx,m in enumerate(cat_attrs) if m in same_attrs.keys()]) if cat_attrs else None

		#ar = np.array([2 ** i for i in range(len(self._attrs))])
		#self._objs_bit = [(g*ar).sum() for g in self._data]
		#del ar

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

		same_attrs = {attrs[k]: np.array(attrs)[v] if len(v)>0 else v for k, v in same_attrs.items()}
		same_objs = {objs[k]: np.array(objs)[v] if len(v)>0 else v for k,v in same_objs.items()}

		return same_objs, same_attrs

	def get_attrs(self):
		return self._attrs_full

	def get_objs(self):
		return self._objs_full

	def get_data(self):
		return self._data_full

	def get_attr_values(self, m):
		if type(m) == int:
			m_idx = m
			if m_idx<0 or m_idx>len(self._attrs)-1:
				raise ValueError(f"There are only {len(self._attrs)} attributes (Suggested {m_idx})")
		elif type(m) == str:
			if m not in self._attrs:
				raise ValueError(f'No such attribute {m}')
			m_idx = np.argmax(self._attrs == m)
		else:
			raise TypeError(f"Possible values for 'm' are string and int type")

		return self._data[:, m_idx]

	def get_obj_values(self, g, trust_mode=False):
		if not trust_mode:
			if type(g) == int:
				g_idx = g
				if g_idx<0 or g_idx>len(self._objs)-1:
					raise ValueError(f"There are only {len(self._objs)} objects (Suggested {g_idx})")
			elif type(g) == str:
				if g not in self._objs:
					raise ValueError(f'No such object {g}')
				g_idx = np.argmax(self._objs == g)
			else:
				raise TypeError(f"Possible values for 'g' are string and int type")
		else:
			g_idx=g

		return self._data[g_idx]

	def get_extent(self, ms, trust_mode=False, verb=True):
		if not trust_mode:
			ms_idxs = []
			ms_error = []
			ms = list(ms) if type(ms) == tuple else \
							[ms] if type(ms) != list else ms
			for m in ms:
				if type(m) == str:
					if m in self._attrs:
						ms_idxs.append(np.argmax(self._attrs==m))
					else:
						ms_error.append(m)
				elif type(m) == int:
					if m >= 0 and m < len(self._attrs):
						ms_idxs.append(m)
					else:
						ms_error.append(m)
				else:
					ms_error.append(m)
			if len(ms_error)>0:
				raise ValueError(f'Wrong attributes are given: {ms_error}')
		else:
			ms_idxs = ms

		#ext = self._get_extent_bitwise(ms_idxs)

		ext = np.arange(len(self._objs))
		ext = list(ext[self._data[:, ms_idxs].sum(1) == len(ms_idxs)])
		ext = [self._objs[g] for g in ext] if verb else ext
		return ext

	def _get_extent_bitwise(self, ms):
		if len(ms) == 0:
			return list(range(len(self._objs)))
		ms_bit = (2 ** np.array(ms)).sum()
		ext_ = [idx for idx, g in enumerate(self._objs_bit) if (g & ms_bit) == ms_bit]
		return ext_

	def get_intent(self, gs, trust_mode=False, verb=True):
		if not trust_mode:
			gs_idxs = []
			gs_error = []
			gs = list(gs) if type(gs) == tuple else \
							[gs] if type(gs) != list else gs
			for g in gs:
				if type(g) == str:
					if g in self._objs:
						gs_idxs.append(np.argmax(self._attrs == g))
					else:
						gs_error.append(g)
				elif type(g) == int:
					if g >= 0 and g < len(self._objs):
						gs_idxs.append(g)
					else:
						gs_error.append(g)
				else:
					gs_error.append(g)
			if len(gs_error)>0:
				raise ValueError(f'Wrong objects are given: {gs_error}')
		else:
			gs_idxs = gs

		#int_ = self._get_intent_bitwise(gs_idxs)
		int_ = np.arange(len(self._attrs))
		cntx = self._data[gs_idxs]
		int_ = list(int_[cntx.sum(0) == len(gs_idxs)])
		int_ = [self._attrs[m] for m in int_] if verb else int_
		return int_

	def _get_intent_bitwise(self, gs):
		if len(gs) == 0:
			return list(range(len(self._attrs)))

		int_ = self._objs_bit[gs[0]]
		for g in gs[1:]:
			int_ &= self._objs_bit[g]
		int_ = [idx for idx, c in enumerate(bin(int_)[2:][::-1]) if c == '1']
		return int_

	def _get_pattern_intersect(self, ms):
		pass

	def get_table(self):
		return pd.DataFrame(self._data, index=self._objs, columns=self._attrs)

	def __repr__(self):
		s = f"Num of objects: {len(self._objs)}, Num of attrs: {len(self._attrs)}\n"
		s += f"Objects: {','.join(self._objs[:5])+',...' if len(self._objs)>5 else ','.join(self._objs)}\n"
		s += f"Attrs: {','.join(self._attrs[:5]) + ',...' if len(self._attrs) > 5 else ','.join(self._attrs)}\n"
		s += self.get_table().head().__repr__()
		return s


class FormalManager:
	def __init__(self, context, ds_obj=None, target_attr=None, cat_feats=None, task_type=None):
		self._context = context
		if ds_obj is not None:
			ds_obj.index = ds_obj.index.astype(str)
		self._ds_obj = ds_obj
		self._concepts = None
		self._target_attr = target_attr
		self._top_concept = None
		self._bottom_concept = None
		self._cat_feats = cat_feats
		self._task_type = task_type

	def get_context(self):
		return self._context

	def get_concepts(self):
		return self._concepts

	def get_concept_by_id(self, id):
		return [c for c in self._concepts if c._idx==id][0]

	def construct_concepts(self, algo='mit', max_iters_num=None, max_num_attrs=None, min_num_objs=None, use_tqdm=True,
						   is_monotonic=False):
		if algo == 'CBO':
			concepts = self._close_by_one(max_iters_num, max_num_attrs, min_num_objs, use_tqdm)
		elif algo == 'mit':
			concepts = self._concepts_by_mit()
		else:
			raise ValueError('The only supported algorithm is CBO (CloseByOne')

		if is_monotonic:
			concepts = {Concept([g for g in self._context._objs if g not in  c._extent], c._intent) for c in concepts}

		new_concepts = set()
		for idx, c in enumerate(sorted(concepts, key=lambda c: (len(c.get_intent()), ','.join(c.get_intent())))):
			ext_short = c.get_extent()
			int_short = c.get_intent()
			ext_ = [g_ for g in ext_short for g_ in [g]+list(self._context._same_objs[g])]
			int_ = [m_ for m in int_short for m_ in [m]+list(self._context._same_attrs[m])]
			pattern = self._find_concept_pattern(c) if self._ds_obj is not None else None
			y_true = self._context._y_true[np.isin(self._context._objs_full, ext_)] if self._context._y_true is not None else None
			y_pred = self._context._y_pred[np.isin(self._context._objs_full, ext_)] if self._context._y_pred is not None else None
			metrics = self._calc_metrics(y_true, y_pred) if len(ext_) > 0 and all([x is not None for x in [y_true, y_pred, self._task_type]]) else None
			y_pred_mean = y_pred.mean() if y_pred is not None and len(y_pred)>0  else None
			y_true_mean = y_true.mean() if y_true is not None and len(y_true) > 0 else None
			new_concepts.add(Concept(ext_, int_, idx=idx, pattern=pattern,
									 y_true_mean=y_true_mean, y_pred_mean=y_pred_mean, metrics=metrics,
									 extent_short=ext_short, intent_short=int_short,
									 is_monotonic=is_monotonic))
		concepts = new_concepts
		self._concepts = concepts

		self._top_concept = [c for c in concepts if len(c.get_extent()) == len(self._context.get_objs())]
		self._top_concept = self._top_concept[0] if len(self._top_concept) > 0 else None

		self._bottom_concept = [c for c in concepts if len(c.get_intent()) == len(self._context.get_attrs())]
		self._bottom_concept = self._bottom_concept[0] if len(self._bottom_concept)>0 else None

	def delete_concept(self, c_idx):
		c = self.get_concept_by_id(c_idx)
		upns, lns = c._up_neighbs, c._low_neighbs
		if upns is None or len(upns) == 0:
			raise KeyError(f'Cannot delete concept {c_idx}. It may be supremum')

		#if lns is None or len(lns) == 0:
		#	raise KeyError(f'Cannot delete concept {c_idx}. It may be infinum')

		for upn_id in upns:
			upn = self.get_concept_by_id(upn_id)
			upn._low_neighbs.remove(c_idx)
			upn._low_neighbs = upn._low_neighbs | lns

		for ln_id in lns:
			ln = self.get_concept_by_id(ln_id)
			ln._up_neighbs.remove(c_idx)
			ln._up_neighbs = ln._up_neighbs | upns
		self._concepts = [c_ for c_ in self._concepts if c_ != c]

		idx_map = {c_._idx: i for i, c_ in
				   enumerate(sorted(self._concepts, key=lambda c_: (len(c_.get_intent()), ','.join(c_.get_intent()))))
				   }
		for c_ in self._concepts:
			c_._idx = idx_map[c_._idx]
			c_._up_neighbs = {idx_map[up_id] for up_id in c_._up_neighbs}
			c_._low_neighbs = {idx_map[ln_id] for ln_id in c_._low_neighbs}

	def _concepts_by_mit(self):
		cntx_mit = concepts_mit.Context(self._context._objs,#f"g{x}" for x in range(len(self._context.get_objs()))],
										self._context._attrs,#[f"m{x}" for x in range(len(self._context.get_attrs()))],
										self._context._data
										)

		self._lattice_mit = cntx_mit.lattice
		concepts = {Concept(ext_, int_) for ext_, int_ in self._lattice_mit}
		return concepts

	def _calc_metrics(self, y_true, y_pred):
		from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
			r2_score, mean_absolute_error, mean_squared_error
		if self._task_type == 'regression':
			ms = {
				'r2': r2_score(y_true, y_pred),
				'me': np.mean(y_true-y_pred),
				'ame': np.abs(np.mean(y_true-y_pred)),
				'mae': mean_absolute_error(y_true, y_pred),
				'mse': mean_squared_error(y_true, y_pred),
				'mape': np.mean(np.abs(y_true-y_pred)/y_true),
			}
		elif self._task_type == 'binary classification':
			ms = {
				'accuracy': round(accuracy_score(y_true, y_pred),2),
				'precision': round(precision_score(y_true, y_pred),2),
				'recall': round(recall_score(y_true, y_pred),2),
				'neg_precision': round(precision_score(1-y_true, 1-y_pred), 2),
				'neg_recall': round(recall_score(1-y_true, 1-y_pred), 2),
			}
		else:
			raise ValueError(f"Given task type {self._task_type} is not supported. Possible values are 'regression', 'binary classification'")
		return ms

	def _close_by_one(self, max_iters_num, max_num_attrs, min_num_objs, use_tqdm):
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
		cncpts_map = {c._idx:c for c in self._concepts}
		all_low_neighbs = {c._idx:set() for c in self._concepts}

		for cncpt_idx in tqdm(range(n_concepts-1, -1, -1), disable=not use_tqdm):
			concept = cncpts_map[cncpt_idx]
			concept._low_neighbs = set()
			possible_neighbs = set(range(cncpt_idx+1, n_concepts))

			while len(possible_neighbs)>0:
				pn_idx = min(possible_neighbs)
				possible_neighbs.remove(pn_idx)

				if cncpts_map[pn_idx].is_subconcept_of(concept):
					all_low_neighbs[cncpt_idx] = all_low_neighbs[cncpt_idx]|{pn_idx}|all_low_neighbs[pn_idx]
					concept._low_neighbs.add(pn_idx)
					possible_neighbs = possible_neighbs-all_low_neighbs[pn_idx]

			concept._up_neighbs = set()
			for ln_idx in concept._low_neighbs:
				cncpts_map[ln_idx]._up_neighbs.add(concept._idx)

	def _find_new_concept_objatr(self):
		cncpt_dict = {c._idx:c for c in self._concepts}
		for c in self._concepts:
			c._new_attrs = tuple(set(c.get_intent())-{m for un_idx in c._up_neighbs for m in cncpt_dict[un_idx].get_intent()})
			c._new_objs = tuple(set(c.get_extent()) - {m for ln_idx in c._low_neighbs for m in cncpt_dict[ln_idx].get_extent()})

	def _calc_concept_levels(self):
		concepts = sorted(self._concepts, key=lambda c: c._idx)
		#concepts = sorted(self._concepts, key=lambda c: (len(c.get_intent()), ','.join(c.get_intent())))
		if self._top_concept is None:
			return
		#concepts_to_check = [self._top_concept]
		concepts[0]._level = 0
		for c in concepts[1:]:
			c._level = max([concepts[un]._level for un in c._up_neighbs]) + 1

	def construct_lattice(self, use_tqdm=False):
		self._construct_lattice_connections(use_tqdm)
		self._calc_concept_levels()
		self._find_new_concept_objatr()

	def get_plotly_fig(self, level_sort=None, sort_by=None, y_precision=None, color_by=None, title=None,
					   cbar_title=None, cmin=None, cmid=None, cmax=None, cmap='RdBu',
					   new_attrs_lim=5, new_objs_lim=5,
					   metrics_to_print='all', figsize=None):
		connections_dict = {}
		for c in self.get_concepts():
			connections_dict[c._idx] = [ln_idx for ln_idx in c._low_neighbs]
		level_widths = {}
		concepts = sorted(self._concepts, key=lambda c: c._idx)
		for c in concepts:
			level_widths[c._level] = level_widths.get(c._level, 0) + 1
		max_width = max(level_widths.values())
		n_levels = len(level_widths)
		pos = {}

		last_level = None
		cur_level_idx = None
		for c in sorted(concepts, key=lambda c: c._level):
			cl = c._level
			cur_level_idx = cur_level_idx + 1 if cl == last_level else 1
			last_level = cl
			pos[c._idx] = (cur_level_idx - level_widths[cl] / 2 - 0.5, n_levels - cl)

		def sort_feature(c, sort_by):
			if sort_by == 'y_true':
				return c._y_true_mean
			if sort_by == 'y_pred':
				return c._y_pred_mean
			if c._metrics is not None:
				for k, v in c._metrics.items():
					if k == sort_by:
						return v
			return None
            
		if level_sort is not None and sort_by is not None:
			level_sort = n_levels//2 if level_sort == 'mean' else level_sort

			cncpt_by_levels = {}
			for c in concepts:
				cncpt_by_levels[c._level] = cncpt_by_levels.get(c._level,[])+[c]
			pos = {}


				#raise ValueError(f'Unknown feature to sort by: {sort_by}')

			if level_sort == 'all':
				for cl in range(0, len(level_widths)):
					for c_l_idx, c in enumerate(sorted(cncpt_by_levels[cl], key=lambda c: sort_feature(c, sort_by))):
						pos[c._idx] = (c_l_idx - level_widths[cl]/2 + 0.5, n_levels - cl)
			else:
				cl = level_sort
				for c_l_idx, c in enumerate(sorted(cncpt_by_levels[level_sort], key=lambda c: sort_feature(c, sort_by))):
					pos[c._idx] = (c_l_idx - level_widths[cl]/2 + 0.5, n_levels - cl)

				for cl in range(level_sort-1,-1,-1):
					for c_l_idx,c in enumerate(cncpt_by_levels[cl]):
						pos[c._idx] = (np.mean([pos[ln][0] for ln in c._low_neighbs if ln in pos]), n_levels-cl)

				for cl in range(level_sort+1,n_levels):
					for c_l_idx,c in enumerate(cncpt_by_levels[cl]):
						pos[c._idx] = (np.mean([pos[un][0] for un in c._up_neighbs if un in pos]), n_levels-cl)

			# center to 0
			for cl in range(n_levels):
				m = np.mean([pos[c._idx][0] for c in cncpt_by_levels[cl]])
				for c_l_idx,c in enumerate(sorted(cncpt_by_levels[cl], key=lambda c: pos[c._idx][0])):
					pos[c._idx] = (pos[c._idx][0] - m, pos[c._idx][1])
					pos[c._idx] = (c_l_idx - level_widths[cl] / 2 + 0.5, n_levels - cl)

		G = nx.from_dict_of_lists(connections_dict)
		nx.set_node_attributes(G, 'pos', {c._idx:pos[c._idx] for c in concepts})

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
				# 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
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
			c = concepts[node]
			node_color.append(sort_feature(c, color_by) if sort_feature(c,color_by) is not None else 'grey')
			#node_color.append(c._mean_y if c._mean_y is not None else 'grey')
			# node_text.append('a\nbc')
			node_text.append(c.pretty_repr(print_mean_y_true=True,
										   print_mean_y_pred=True,
										   metrics_to_print=metrics_to_print,
										   y_precision=y_precision).replace('\n', '<br>')+\
							 '')#f'\npos({pos[c._idx]})')
			new_attrs_str = '' if len(c._new_attrs) == 0 else\
				f"{','.join(c._new_attrs) if c._new_attrs else ''}" if len(c._new_attrs)<new_attrs_lim \
					else f"n: {len(c._new_attrs)}"
			new_objs_str = '' if len(c._new_objs) == 0 else \
				f"{','.join(c._new_objs) if c._new_objs else ''}" if len(c._new_objs)<new_objs_lim \
					else f"n: {len(c._new_objs)}"
			node_title.append(new_attrs_str+'<br>'+new_objs_str)
		# node_text.append('# of connections: '+str(len(adjacencies[1])))

		node_trace.marker.color = node_color
		node_trace.hovertext = node_text
		node_trace.text = node_title

		if cmin is not None:
			node_trace.marker.cmin = cmin
		if cmax is not None:
			node_trace.marker.cmax = cmax
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
							width = figsize[0] if figsize is not None else 1000,
							height = figsize[1] if figsize is not None else 500,
						)
						)
		return fig

	def _find_concept_pattern(self, cncpt):
		ds = self._ds_obj
		cat_feats = self._cat_feats
		cat_feats = cat_feats if cat_feats else []

		if cncpt._extent is None or cncpt._intent is None:
			return None
		attrs = cncpt._intent
		if not all([f in ds.columns for f in attrs]):
			attrs = list(set([f.split('__')[0] for f in attrs]))

		cds = ds.loc[cncpt._extent, attrs]
		patrn = {}
		for f in attrs:
			if f in cat_feats or ds[f].dtype == bool:
				s = cds[f].value_counts().sort_values(ascending=False)
				s /= len(cds)
				s = s.round(2)
				patrn[f] = s.to_dict()
			else:
				patrn[f] = (cds[f].min(), cds[f].median(), cds[f].max())
		return patrn