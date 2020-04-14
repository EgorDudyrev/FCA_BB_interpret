import numpy as np
import pandas as pd

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
            objs = list(range(len(data.shape[1]))) if objs is None else objs
            attrs = list(range(len(objs[0]))) if attrs is None else attrs
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

    def get_table(self, is_full=True):
        return pd.DataFrame(self._data, index=self._objs_full if is_full else self._objs,
                            columns=self._attrs_full if is_full else self._attrs)
