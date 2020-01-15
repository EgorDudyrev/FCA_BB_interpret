import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import concepts as concepts_mit


class Concept:
	def __init__(self, extent, intent, idx=None, title=None):
		self._extent = np.array(extent)
		self._intent = np.array(intent)
		self._low_neighbs = None
		self._up_neighbs = None
		self._idx = idx
		self._title = title
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
		s += f"extent (len: {len(self._extent)}): {','.join([f'{g}' for g in self._extent]) if len(self._extent)>0 else 'emptyset'}\n"
		s += f"intent (len: {len(self._intent)}): {','.join([f'{m}' for m in self._intent]) if len(self._intent)>0 else 'emptyset'}\n"
		if self._low_neighbs is not None:
			s += f"lower neighbours (len: {len(self._low_neighbs)}): " + \
				 f"{','.join([f'{c}' for c in self._low_neighbs]) if len(self._low_neighbs)>0 else 'emptyset'}\n"
		if self._up_neighbs is not None:
			s += f"upper neighbours (len: {len(self._up_neighbs)}): " + \
				 f"{','.join([f'{c}' for c in self._up_neighbs]) if len(self._up_neighbs)>0 else 'emptyset'}\n"
		return s

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
		return all([g in c._extent for g in self._extent]) and all([m in self._intent for m in c._intent])
		#return self._is_subconcept_of_bit(c)

	def _is_subconcept_of_bit(self, c):
		return (self._extent_bit & c._extent_bit == self._extent_bit)\
			   and (c._intent_bit & self._intent_bit == c._intent_bit)


class Context:
	def __init__(self, data, objs=None, attrs=None):
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

		assert data.dtype == bool, 'Only Boolean contexts are supported for now'

		self._data = data
		self._objs = np.array(objs)
		self._attrs = np.array(attrs)

		#ar = np.array([2 ** i for i in range(len(self._attrs))])
		#self._objs_bit = [(g*ar).sum() for g in self._data]
		#del ar

	def get_attrs(self):
		return self._attrs

	def get_objs(self):
		return self._objs

	def get_data(self):
		return self._data

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

	def get_table(self):
		return pd.DataFrame(self._data, index=self._objs, columns=self._attrs)

	def __repr__(self):
		s = f"Num of objects: {len(self._objs)}, Num of attrs: {len(self._attrs)}\n"
		s += f"Objects: {','.join(self._objs[:5])+',...' if len(self._objs)>5 else ','.join(self._objs)}\n"
		s += f"Attrs: {','.join(self._attrs[:5]) + ',...' if len(self._attrs) > 5 else ','.join(self._attrs)}\n"
		s += self.get_table().head().__repr__()
		return s


class FormalManager:
	def __init__(self, context, target_attr=None):
		self._context = context
		self._concepts = None
		self._target_attr = target_attr
		self._top_concept = None
		self._bottom_concept = None

	def get_context(self):
		return self._context

	def get_concepts(self):
		return self._concepts

	def construct_concepts(self, algo='mit', max_iters_num=None, max_num_attrs=None, min_num_objs=None, use_tqdm=True):
		if algo == 'CBO':
			concepts = self._close_by_one(max_iters_num, max_num_attrs, min_num_objs, use_tqdm)
		elif algo == 'mit':
			concepts = self._concepts_by_mit()
		else:
			raise ValueError('The only supported algorithm is CBO (CloseByOne')

		concepts = set([Concept(c.get_extent(), c.get_intent(), idx)
				for idx, c in enumerate(sorted(concepts, key=lambda x: len(x.get_intent())))])
		self._concepts = concepts

		self._top_concept = [c for c in concepts if len(c.get_extent()) == len(self._context.get_objs())][0]
		self._bottom_concept = [c for c in concepts if len(c.get_intent()) == len(self._context.get_attrs())][0]

	def _concepts_by_mit(self):
		cntx_mit = concepts_mit.Context([f"g{x}" for x in range(len(self._context.get_objs()))],
										[f"m{x}" for x in range(len(self._context.get_attrs()))],
										self._context.get_data()
										)

		concepts = {Concept(ext_, int_) for ext_, int_ in cntx_mit.lattice}
		return concepts

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

	def construct_lattice_connections(self, use_tqdm=True):
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