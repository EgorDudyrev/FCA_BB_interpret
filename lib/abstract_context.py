import numpy as np
import pandas as pd

from .utils_ import get_not_none, repr_set

class AbstractConcept:
    def __init__(self, extent, intent, idx=None, title=None,
                 metrics=None, extent_short=None, intent_short=None,
                 is_monotonic=False):
        #self._extent = extent
        self._extent = list(extent)
        self._intent = intent
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
        s = self.__class__.__name__
        s += f" {self._idx}" if self._idx is not None else ''
        s += f" {self._title}" if self._title is not None else ''
        s += f"\n"
        s += f"level: {self._level}" if print_level and self._level is not None else ''
        s += '\n'
        return s

    @staticmethod
    def _get_intent_as_array(int_):
        return int_

    def __repr__(self):
        s = self._repr_concept_header()

        for set_, set_name in [(self._extent, 'extent'),
                               (self._get_intent_as_array(self._intent), 'intent'),
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
                  (self._get_intent_as_array(self._intent), 'intent', True),
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
        raise NotImplementedError


class AbstractContext:
    def __init__(self, data, objs=None, attrs=None, y_true=None, y_pred=None):
        pass

    def _check_input_data(self, data, objs, attrs, y_true, y_pred):
        if type(data) == list:
            data = np.array(data)

        if type(data) == pd.DataFrame:
            objs = list(data.index) if objs is None else objs
            attrs = list(data.columns) if attrs is None else attrs
            data = data.values
        elif type(data) == np.ndarray:
            objs = list(range(data.shape[0])) if objs is None else objs
            attrs = list(range(data.shape[1])) if attrs is None else attrs
        else:
            raise TypeError(f"DataType {type(data)} is not understood. np.ndarray or pandas.DataFrame is required")

        objs = np.array([str(g) for g in objs])
        attrs = np.array([str(m) for m in attrs])

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
            assert len(y_pred) == len(
                data), f'Data and Y_vals have different num of objects ( Data: {len(data)}, y_vals: {len(y_pred)})'
        else:
            self._y_pred = None

        return data, objs, attrs, y_true, y_pred

    def get_attrs(self, is_full=True):
        return self._attrs_full if is_full else self._attrs

    def get_objs(self, is_full=True):
        return self._objs_full if is_full else self._objs

    def get_data(self):
        return self._data_full

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
        elif type(x) in [str, np.str_]:
            x = np.str_(x)
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

    def get_table(self, is_full=True):
        return pd.DataFrame(self._data, index=self._objs_full if is_full else self._objs,
                            columns=self._attrs_full if is_full else self._attrs)
