"""
MIT License

Copyright (c) 2017 Friedrich Hagedorn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
import inspect, ast
import itertools
from textwrap import dedent
from time import sleep
from sortedcontainers import SortedDict, SortedSet
from collections import OrderedDict, namedtuple
from functools import wraps
import json

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import colorConverter
from colorsys import rgb_to_hls, hls_to_rgb

try:
    from itertools import zip_longest
except AttributeError:
    from itertools import izip_longest as zip_longest


def read_json(s):
    def decode(dct):
        if '__class__' in dct:
            name = dct.pop('__class__')
            #~ print('{}: {}'.format(name, dct))
            objs = dict(
                Sweep=Sweep,
                Iterate=Iterate,
                BisectSweep=BisectSweep,
                Concat=Concat,
                Param=Param,
                Parameters=Parameters,
                Task=Task,
                TaskSweep=TaskSweep,
                LinePlot=LinePlot,
                SweepPlot=SweepPlot,
            )
            return objs[name]._from_dict(dct)
        return dct
    return json.loads(s, object_hook=decode)


def write_json(obj):
    tw = TabWriter()
    return tw.to_json(obj)


class Sweep(object):
    def __init__(self, start, stop, step=1, num=None, endpoint=True):
        self.start = start
        self.stop = stop
        self.endpoint = endpoint

        delta = self.stop - self.start
        if num is None:
            self.step = abs(step) if delta > 0 else -abs(step)
            num = round(delta / self.step)
            self.num = num + 1 if self.endpoint else num
        else:
            self.num = num
            div = (self.num - 1) if self.endpoint else self.num
            div = div if div > 1 else 1
            self.step = float(delta) / div
        self._state_stop = self.num - 1
        self.reset()

    def reset(self):
        self.state = 0
        self.is_iterating = False

    def __repr__(self):
        args = []
        args.append(repr(self.start))
        args.append(repr(self.stop))
        for name in ['step']:
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        if not self.endpoint:
            name = 'endpoint'
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['start'] = self.start
        dct['stop'] = self.stop
        dct['step'] = self.step
        if not self.endpoint:
            dct['endpoint'] = self.endpoint
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)

    @property
    def min(self):
        return min(self.start, self.stop)

    @property
    def max(self):
        return max(self.start, self.stop)

    def next_state(self):
        self.state += 1

    def out(self):
        self.is_iterating = True
        return self.start + self.step*self.state

    def is_running(self):
        return 0 <= self.state < self._state_stop

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        return self.num

    @property
    def idx(self):
        return self.state


class Iterate(object):
    def __init__(self, *items):
        self.items = list(items)
        self.reset()

    def reset(self):
        self.idx = 0
        self.is_iterating = False

    @property
    def min(self):
        try:
            return min(self.items)
        except Exeption:
            return None

    @property
    def max(self):
        try:
            return max(self.items)
        except Exeption:
            return None

    @property
    def item(self):
        return self.items[self.idx]

    @property
    def state(self):
        return self.idx

    @state.setter
    def state(self, value):
        self.idx = value
        self.is_iterating = False

    def next_state(self):
        self.idx += 1
        return self.idx

    def out(self):
        self.is_iterating = True
        return self.item

    @property
    def _state_stop(self):
        return len(self) - 1

    def is_running(self):
        return self.idx < self._state_stop

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        args = []
        for item in self.items:
            arg = '{}'.format(repr(item))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['items'] = self.items
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(*dct['items'])


class Zip(object):
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)
        self.reset()

    def reset(self):
        self.idx = 0
        self.is_iterating = False
        for sweep in self.sweeps:
            sweep.reset()

    @property
    def min(self):
        return None
        try:
            return tuple(sweep.min for sweep in self.sweeps)
        except Exception:
            return None

    @property
    def max(self):
        return None
        try:
            return tuple(sweep.max for sweep in self.sweeps)
        except Exception:
            return None

    @property
    def state(self):
        return self.idx

    @state.setter
    def state(self, value):
        self.idx = value

    def next_state(self):
        self.idx += 1
        for sweep in self.sweeps:
            sweep.next_state()

    def out(self):
        self.is_iterating = True
        return tuple(sweep.out() for sweep in self.sweeps)

    @property
    def _state_stop(self):
        return len(self) - 1

    def is_running(self):
        return all(sweep.is_running() for sweep in self.sweeps)

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        return min(len(sweep) for sweep in self.sweeps)

    def __repr__(self):
        args = []
        for sweep in self.sweeps:
            arg = '{!r}'.format(sweep)
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['sweeps'] = self.sweeps
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(*dct['sweeps'])


class Concat:
    def __init__(self, *sweeps):
        self.iterate = Iterate(*sweeps)
        self.reset()

    def append(self, sweep):
        self.iterate.items.append(sweep)
        self.reset()

    def reset(self):
        for sweep in self.iterate.items:
            sweep.reset()
        self.iterate.reset()
        self.is_iterating = False

    @property
    def min(self):
        return min(sweep.min for sweep in self.iterate.items)

    @property
    def max(self):
        return max(sweep.max for sweep in self.iterate.items)

    def out(self):
        self.is_iterating = True
        return self.iterate.out().out()

    @property
    def sweeps(self):
        return self.iterate.out(), self.iterate

    @property
    def state(self):
        state_lengths = []
        for sweep in self.iterate.items:
            try:
                state_lengths.append(len(sweep.state))
            except TypeError:
                state_lengths.append(1)
        maxlen = max(state_lengths) + 1
        state = [self.iterate.state]
        try:
            state.extend(self.iterate.out().state)
        except TypeError:
            state.append(self.iterate.out().state)
        state.extend([None]*(maxlen - len(state)))
        return tuple(state)

    @state.setter
    def state(self, values):
        self.iterate.state = values[0]
        try:
            N = len(self.iterate.out().state)
            self.iterate.out().state = values[1:N+1]
        except TypeError:
            self.iterate.out().state = values[1]

    def next_state(self):
        idx = 0
        sweeps = self.sweeps
        while sweeps[idx].is_finished():
            idx += 1
            if idx >= len(sweeps):
                return
        sweeps[idx].next_state()
        while idx > 0:
            idx -= 1
            self.sweeps[idx].reset()

    def is_running(self):
        return any(sweep.is_running() for sweep in self.sweeps)

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        return sum(len(sweep) for sweep in self.iterate.items)

    @property
    def idx(self):
        sweeps = self.iterate.items[:self.iterate.idx + 1]
        return sum(sweep.idx for sweep in sweeps) + self.iterate.idx

    def __repr__(self):
        args = []
        for item in self.iterate.items:
            arg = '{}'.format(repr(item))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['sweeps'] = self.iterate.items
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(*dct['sweeps'])


class BisectSweep(object):
    def __init__(self, start, stop, cycles=4):
        self.start = start
        self.stop = stop
        self.cycles = cycles
        nums = [2**n + 1 for n in range(cycles)]
        self.__len__ = nums[-1]
        steps = []
        for num in nums:
            div = num - 1
            div = div if div > 1 else 1
            step = float(stop - start) / div
            steps.append(step)
        self._sweeps = []
        for n in range(len(steps)):
            endpoint = False if n else True
            sweeps = [Sweep(start, stop, steps[0], endpoint=endpoint)]
            last_step = steps[0]
            _steps = steps[1:n+1]
            for _n, step in enumerate(_steps):
                n0 = step if _n+1 == len(_steps) else 0
                sweep = Sweep(n0, last_step, step, endpoint=False)
                sweeps.append(sweep)
                last_step = step
            self._sweeps.append(sweeps[::-1])
        self.index = Sweep(0, len(steps)-1)
        self.reset()

    def reset(self):
        for sweep in self.all_sweeps:
            sweep.reset()
        self.is_iterating = False

    @property
    def min(self):
        return self._sweeps[0][0].min

    @property
    def max(self):
        return self._sweeps[0][0].max

    @property
    def step_sweeps(self):
        idx = self.index.out()
        return self._sweeps[idx]

    @property
    def all_sweeps(self):
        return self.step_sweeps + [self.index]

    def out(self):
        self.is_iterating = True
        return sum(sweep.out() for sweep in self.step_sweeps)

    @property
    def state(self):
        return tuple(sweep.state for sweep in self.all_sweeps)

    def next_state(self):
        sweeps = self.all_sweeps
        idx = 0
        while sweeps[idx].is_finished():
            idx += 1
            if idx >= len(sweeps):
                return
        sweeps[idx].next_state()
        while idx > 0:
            idx -= 1
            sweeps[idx].reset()

    def is_running(self):
        return any(sweep.is_running() for sweep in self.all_sweeps)

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    @property
    def idx(self):
        sweeps = self.iterate.items[:self.iterate.idx + 1]
        return sum(sweep.idx for sweep in sweeps) + self.iterate.idx

    def __repr__(self):
        args = []
        args.append(repr(self.start))
        args.append(repr(self.stop))
        for name in ['cycles']:
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['start'] = self.start
        dct['stop'] = self.stop
        dct['cycles'] = self.cycles
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)


def bisect_sweep(start, stop, cycles=4):
    delta = abs(stop - start)
    sweeps = [Sweep(start, stop, step=delta)]
    offs = delta / 2.0
    for n in range(1, cycles):
        sweep = Sweep(start + offs, stop, step=2*offs)
        sweeps.append(sweep)
        offs = offs / 2.0
    return Concat(*sweeps)


class Dependency(object):
    def __init__(self, obj, key=None):
        self.obj = obj
        self.key = key

    def eval(self):
        if self.key is None:
            return self.obj.out()
        try:
            return self.obj.get(self.key)
        except AttributeError:
            return getattr(self.obj, self.key)
    out = eval

    @property
    def state(self):
        try:
            return self.obj.state
        except AttributeError:
            return None

    def next_state(self):
        return self.obj.next_state()

    def __repr__(self):
        #~ return '<dependency>'
        if self.key is None:
            return 'Dependency({!r})'.format(self.obj)
        else:
            return 'Dependency({!r}, key={!r})'.format(self.obj, self.key)

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['obj'] = self.obj
        dct['key'] = self.key
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)


class ParamsDep(object):
    def __init__(self, params, key=None):
        self.params = params
        self.key = key

    def out(self):
        return self.params[self.key].out()

    @property
    def state(self):
        return self.params[self.key].value.state

    @property
    def min(self):
        return self.params[self.key].value.min

    @property
    def max(self):
        return self.params[self.key].value.max

    @property
    def idx(self):
        return self.params[self.key].value.idx

    def __len__(self):
        return self.params[self.key].value.__len__()

    def __repr__(self):
        return 'ParamsDep({!r})'.format(self.key)

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['obj'] = self.obj
        dct['key'] = self.key
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)


def config(func=None, delayed=False):
    def myconfig(_func):
        @wraps(_func)
        def myfunc(self, *args, **kwargs):
            task = self if isinstance(self, Task) else self._parent._parent
            task._config.append( (_func, self, args, kwargs) )
            if delayed:
                task._configure_idx.append(len(task._config) - 1)
            else:
                _func(self, *args, **kwargs)
        return myfunc
    if func:
        return myconfig(func)
    else:
        return myconfig


class Param(object):
    def __init__(self, name, default=None, value=None, key=None, _parent=None):
        self._parent = _parent
        self._name = name
        self._default = default
        self.value = value if value is not None else default
        self.key = key

    @property
    def name(self):
        return self._name

    @property
    def default(self):
        return self._default

    def out(self):
        try:
            if self.key is not None:
                return self._value.out()[self.key]
            else:
                return self._value.out()
        except AttributeError:
            return self._value

    @property
    def value(self):
        return self._value

    # this should be @config but it's often used internally (change API)
    @value.setter
    def value(self, value):
        self._value = value
        self.key = None
        if self._parent is not None:
            param = self._parent._params.pop(self.name)
            self._parent._params[self.name] = param

    @config
    def use_default(self):
        self.value = self.default
        return self.value

    @config
    def depends_on_param(self, obj):
        self.value = ParamsDep(obj._parent, obj.name)
        #~ return self.value

    @config
    def sweep(self, *args, zip='', **kwargs):
        if isinstance(self.value, (Sweep, Iterate)):
            self.value = Concat(self.value)
        try:
            self.value.append(Sweep(*args, **kwargs))
        except AttributeError:
            self.value = Sweep(*args, **kwargs)
        if zip:
            self.zip(zip)
        #~ return self.value

    @config
    def iterate(self, *args, zip='', **kwargs):
        if isinstance(self.value, (Sweep, Iterate)):
            self.value = Concat(self.value)
        try:
            self.value.append(Iterate(*args, **kwargs))
        except AttributeError:
            self.value = Iterate(*args, **kwargs)
        if zip:
            self.zip(zip)
        #~ return self.value

    @config(delayed=True)
    def plot(self, x='', y='', row=0, col=0, xlabel=None, ylabel=None,
             slices={}, sub_index=[], **kwargs):
        sweep = self.value
        params = self._parent
        state_names = params._state_names
        idx_slices = {}
        if isinstance(sub_index, (list, tuple)):
            subidx = tuple(sub_index)
        else:
            subidx = (sub_index,)
        for param_name, idxs in slices.items():
            for idx, names in enumerate(state_names):
                if param_name in names:
                    if not isinstance(idxs, (list, tuple, Slice)):
                        idxs = [idxs]
                    idx_slices[idx] = idxs
                    break
        for idx, names in enumerate(state_names):
            if self.name in names:
                xlabel = x if x else self.name
                xlim = sweep.min, sweep.max
                idxs = (idx,) + subidx
                # test if states in sub_index exists in params._states
                sub_states = tuple(params._states)
                for _idx in idxs:
                    try:
                        sub_states = sub_states[_idx]
                    except Exception:
                        msg = "Could not access index {!r} of state {!r}!"
                        msg += "\nMaybe remove 'sub_index' argument of command"
                        msg += " '{}.params.{}.plot'"
                        msg = msg.format(idxs, tuple(params._states),
                                         params._parent.func.__name__,
                                         self.name)
                        raise ValueError(msg)
                params._pm.add_sweep_plot(self.name, idxs, x, y, row, col,
                    xlim=xlim, xlabel=xlabel, ylabel=ylabel,
                    slices=idx_slices, **kwargs)
                break

    @config
    def linesweep(self, x='', y='', row=0, col=0, xlabel=None, ylabel=None,
                  slices={}, **kwargs):
        sweep = self.value
        params = self._parent
        state_names = params._state_names
        idx_slices = {}
        for param_name, idxs in slices.items():
            for idx, names in enumerate(state_names):
                if param_name in names:
                    if not isinstance(idxs, (list, tuple, Slice)):
                        idxs = [idxs]
                    idx_slices[idx] = idxs
                    break
        for idx, names in enumerate(state_names):
            if self.name in names:
                xlabel = x if x else self.name
                xlim = sweep.min, sweep.max
                params._pm.add_linesweep_plot(self.name, idx, x, y, row, col,
                    xlim=xlim, xlabel=xlabel, ylabel=ylabel,
                    slices=idx_slices, **kwargs)
                break

    @config
    def zip(self, names):
        params = self._parent
        self.value = Zip(self.value)
        self.key = 0
        if isinstance(names, str):
            names = [names]
        for n, name in enumerate(names):
            other = getattr(params, name)
            self.value.sweeps.append(other.value)
            other._value = self.value
            other.key = n + 1

    @config
    def interrupt(self, *idx, start=None, stop=None, step=None):
        task = self._parent._parent
        if idx:
            task._interrupts[self] = idx
        else:
            task._interrupts[self] = Slice(start, stop, step)

    def __repr__(self):
        args = []
        names = ['value', 'default', 'name']
        if self.key is not None:
            names.insert(1, 'key')
        for name in names:
            arg = '{}={!r}'.format(name, getattr(self, name))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['name'] = self.name
        dct['default'] = self.default
        dct['value'] = self.value
        dct['key'] = self.key
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(**dct)


class Parameters(object):
    # read-only attributes except for underscore
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError("can't set attribute {}".format(repr(name)))
        object.__setattr__(self, name, value)

    def __init__(self, parent=None):
        self._params = OrderedDict()
        self._parent = parent
        self._pm = PlotManager(task=self._parent)

    def _append(self, param):
        _setattr = object.__setattr__.__get__(self, self.__class__)
        _setattr(param.name, param)
        param._parent = self
        if param.name in self._params:
            self._params.pop(param.name)
        self._params[param.name] = param

    def __iter__(self):
        for name, param in self._params.items():
            yield name, param

    @property
    def _args(self):
        args = OrderedDict()
        for name, param in self:
            args[name] = param.out()
        return args

    @property
    def _arg_names(self):
        for names, sweep in zip(self._sweep_names, self._sweeps):
            name = '_'.join(names)
            yield 'arg_{}'.format(name)
            if isinstance(sweep.state, (tuple, list)):
                for n, _ in enumerate(sweep.state):
                    yield 'idx_{}_{}'.format(name, n)
            else:
                yield 'idx_{}'.format(name)

    @property
    def _arg_values(self):
        for names, sweep in zip(self._sweep_names, self._sweeps):
            name = '_'.join(names)
            yield sweep.out()
            if isinstance(sweep.state, (tuple, list)):
                for state in sweep.state:
                    yield state
            else:
                yield sweep.state

    @property
    def _sweep_names(self):
        sweeps = OrderedDict()
        for name, param in self:
            if hasattr(param.value, 'next_state'):
                sweep = param.value
                names = sweeps.setdefault(sweep, [])
                names.append(name)
        return tuple(sweeps.values())

    @property
    def _sweep_values(self):
        for sweep in self._sweeps:
            yield sweep.out()

    @property
    def _sweeps(self):
        sweeps = set()
        for _, param in self:
            if hasattr(param.value, 'next_state'):
                sweep = param.value
                if sweep not in sweeps:
                    sweeps.add(sweep)
                    yield sweep

    @property
    def _states(self):
        sweeps = set()
        for _, param in self:
            if hasattr(param.value, 'state'):
                sweep = param.value
                if sweep not in sweeps:
                    sweeps.add(sweep)
                    yield sweep.state

    @property
    def _state_names(self):
        objs = OrderedDict()
        for name, param in self:
            if hasattr(param.value, 'state'):
                obj = param.value
                names = objs.setdefault(obj, [])
                names.append(name)
        return tuple(objs.values())

    @property
    def _state_dict(self):
        objs = OrderedDict()
        for name, param in self:
            obj = param.value
            if hasattr(obj, 'state'):
                objs.setdefault(obj, []).append(name)
        states = OrderedDict()
        for obj, names in objs.items():
            name = '_'.join(names)
            try:
                for n, state in enumerate(obj.state):
                    key = '{}_{}'.format(name, n)
                    states[key] = state
            except TypeError:
                states[name] = obj.state
        return states

    @property
    def _idx(self):
        sweeps = tuple(self._sweeps)
        idx = sweeps[0].idx
        num = len(sweeps[0])
        for n in range(1, len(sweeps)):
            idx += sweeps[n].idx * num
            num *= len(sweeps[n])
        return idx

    def _is_running(self):
        return any(sweep.is_running() for sweep in self._sweeps)

    def _is_finished(self):
        return not self._is_running()

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return (name in self._params)

    def _get(self, name, default_value=None):
        return self[name] if name in self else default_value

    def __len__(self):
        return len(self._params)

    def __repr__(self):
        maxlen = max([0] + [len(name) for name in self._params])
        fmtstr = '{{:>{maxlen}}} = {{!r}}'.format(maxlen=maxlen)
        params = (fmtstr.format(name, param.value) for name, param in self)
        params = ',\n    '.join(params)
        return '<Parameters:\n    {}>'.format(params)

        classname = self.__class__.__name__
        args = (',\n' + ' ' * (len(classname) + 1)).join(params)
        return "{classname}({args})".format(classname=classname, args=args)


def return_values(func):
    names = []
    tree = ast.parse(dedent(inspect.getsource(func)))
    for exp in ast.walk(tree):
        if isinstance(exp, ast.FunctionDef):
            break
    for e in exp.body:
        if isinstance(e, ast.Return):
            break
    if not isinstance(e, ast.Return):
        return tuple()
    v = e.value
    if isinstance(v, (ast.Tuple, ast.List)):
        for idx, item in enumerate(v.elts):
            if isinstance(item, ast.Name):
                names.append(item.id)
            else:
                names.append('out_' + str(idx))
    elif isinstance(v, ast.Dict):
        for idx, item in enumerate(v.keys):
            if isinstance(item, ast.Str):
                names.append(item.s)
            else:
                names.append('out_' + str(idx))
    elif isinstance(v, ast.Call) and v.func.id == 'dict' and v.keywords:
        for idx, item in enumerate(v.keywords):
            names.append(item.arg)
    elif isinstance(v, ast.Name):
        names.append(v.id)
    elif v is not None:
        names.append('out')
    return tuple(names)


def namedresult(func):
    result = namedtuple('result' , return_values(func))
    @wraps(func)
    def myfunc(*args, **kwargs):
        return result(*func(*args, **kwargs))
    return myfunc


class Task(object):
    def __init__(self, func):
        self.func = func
        if sys.version_info.major < 3:
            sig = inspect.getargspec(func)
            defaults = dict(zip_longest(sig.args[::-1], sig.defaults[::-1]))
        else:
            defaults = {}
            for name, param in inspect.signature(func).parameters.items():
                if (param.kind is not inspect.Parameter.VAR_POSITIONAL and
                    param.kind is not inspect.Parameter.VAR_KEYWORD):
                    if param.default is param.empty:
                        defaults[name] = None
                    else:
                        defaults[name] = param.default
        self.params = Parameters(parent=self)
        for name, default in defaults.items():
            self.params._append(Param(name, default))
        self._returns = return_values(func)
        common = set(self.params._params).intersection(self._returns)
        if 0 and common:
            msg = 'Common variable names in args and returns: {}'
            common = ['{!r}'.format(name) for name in common]
            msg = msg.format(', '.join(common))
            raise ValueError(msg)
        self._results = OrderedDict()
        self._levels = {}
        self.is_iterating = False
        self._interrupts = {}
        self.is_configured = False
        self._config = []
        self._configure_idx = []

    def configure(self):
        for idx in self._configure_idx:
            func, obj, args, kwargs = self._config[idx]
            func(obj, *args, **kwargs)
        if self.params._pm.am is not None:
            self.params._pm.configure()

    def __repr__(self):
        return '<{}>'.format(self.func.__name__)
        #~ return '<Task {}>'.format(self.func.__name__)

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['name'] = self.func.__name__
        dct['func'] = '<some code in file...>'
        dct['defaults'] = {name: param.default for name, param in self.params}
        config = []
        for func, obj, args, kwargs in self._config:
            conf = OrderedDict()
            if isinstance(obj, Task):
                conf['cmd'] = func.__name__
            else:
                conf['cmd'] = 'params.{}.{}'.format(obj.name, func.__name__)
            if args:
                conf['args'] = args
            if kwargs:
                conf['kwargs'] = kwargs
            config.append(conf)
        dct['config'] = config
        return dct

    @classmethod
    def _from_dict(cls, dct):
        def func(*args, **kwargs):
            return None
        func.__name__ = dct['name']
        task = cls(func)
        for name, default in dct['defaults'].items():
            task.params._append(Param(name, default))
        for conf in dct['config']:
            names = conf['cmd'].split('.')
            if len(names) == 1:
                func = getattr(task, names[0])
            else:
                param = getattr(task.params, names[1])
                func = getattr(param, names[2])
            args = conf.get('args', ())
            kwargs = conf.get('kwargs', {})
            func(*args, **kwargs)
        return task

    def to_json(self):
        return write_json(self)

    @classmethod
    def from_json(cls, s):
        return read_json(s)

    def __str__(self):
        return self.func.__name__

    @property
    def args(self):
        return self.params._args

    @config
    def depends_on_task(self, task, params={}, sweeps=[], squeeze=[]):
        s = ResultSweep(task, list(params.values()), sweeps, squeeze)
        for target, source in params.items():
            param = getattr(self.params, target)
            param.value = s
            param.key = source

    @config
    def plot(self, x='', y='', row=0, col=0, dx='', xoffs='', **kwargs):
        self.params._pm.add_line_plot(x, y, row, col, dx, xoffs, **kwargs)

    def call(self):
        if not self.is_configured:
            self.configure()
            self.is_configured = True
        args = self.args
        result = self.func(**args)
        if not isinstance(result, (tuple, list, dict, OrderedDict)):
            result = [result]
        if isinstance(result, (tuple, list)):
            result = {n: v for n, v in zip(self._returns, result)}
        self._results[tuple(self.params._states)] = args, result
        for names, state in zip(self.params._state_names, self.params._states):
            name = '_'.join(names)
            level = self._levels.setdefault(name, [])
            if state not in level:
                level.append(state)
        return result

    def get_value_names(self, result=None):
        if result is None:
            args, result = self._results[tuple(self.params._states)]
        names = [k for k in sorted(result) if not k.startswith('_')]
        names.extend(k for k in sorted(result) if k.startswith('_'))
        for name in names:
            value = result[name]
            if isinstance(value, np.ndarray):
                if len(value.shape) > 1:
                    shape = '_'.join(str(n) for n in value.shape)
                    fmt = '{} {}_{}'
                    yield fmt.format(name, value.dtype, shape)
                else:
                    yield '{} {}'.format(name, value.dtype)
            elif isinstance(value, complex):
                yield '{} complex'.format(name)
            else:
                yield name

    def get_values(self, result=None):
        if result is None:
            args, result = self._results[tuple(self.params._states)]
        names = [k for k in sorted(result) if not k.startswith('_')]
        names.extend(k for k in sorted(result) if k.startswith('_'))
        for name in names:
            value = result[name]
            if isinstance(value, np.ndarray):
                yield '"{}"'.format(value.tolist())
            else:
                yield value

    def save(self, fname=''):
        if not fname:
            fname = self.func.__name__
        file = open(fname + '.csv', 'w')
        for line in self.to_json().split('\n'):
            file.write('## {}\n'.format(line))
        if self._results:
            header = tuple(self.params._arg_names)
            header += tuple(self.get_value_names())
            file.write(', '.join(header))
            file.write('\n')
            for states, (args, results) in self._results.items():
                line = []
                for names, state in zip(self.params._state_names, states):
                    name = '_'.join(names)
                    line.append(str(args[name]))
                    if isinstance(state, (tuple, list)):
                        line.extend(str(val) for val in state)
                    else:
                        line.append(str(state))
                line.extend(str(val) for val in self.get_values(results))
                file.write(', '.join(line))
                file.write('\n')
        file.close()

    @classmethod
    def load(cls, fname):
        file = open(fname)
        jlines = []
        for line in file:
            if line.startswith('##'):
                jlines.append(line.strip('#'))
            else:
                break
        dtypes = {}
        columns = []
        for col in line.split(','):
            fields = col.split()
            name = fields[0]
            columns.append(name)
            dtype = fields[1:]
            if dtype:
                dtype = dtype[0].split('_')
                dtypes[name] = dtype[0], tuple(int(s) for s in dtype[1:])
        df = pd.read_csv(file,
                         header=None,
                         names=columns,
                         skipinitialspace=True,
                         comment='#')
        task = read_json(''.join(jlines))
        idx_names = []
        for names, _state in zip(task.params._state_names, task.params._states):
            name = '_'.join(names)
            if isinstance(_state, (list, tuple)):
                state_components = []
                for n, _ in enumerate(_state):
                    state_components.append('idx_{}_{}'.format(name, n))
                idx_names.append(state_components)
            else:
                idx_names.append('idx_{}'.format(name))
        state_index = OrderedDict()
        for n in range(len(df)):
            state = ()
            for name in idx_names:
                if isinstance(name, (list, tuple)):
                    state += (tuple(df[_name][n] for _name in name),)
                else:
                    state += (df[name][n],)
            state_index[state] = n

        value_names = []
        for name in columns:
            if not name.startswith('arg_') and not name.startswith('idx_'):
                value_names.append(name)

        def func(*args, **kwargs):
            idx = state_index[tuple(task.params._states)]
            values = df.iloc[idx]
            result = {}
            for name in value_names:
                if name in dtypes:
                    dtype, shape = dtypes[name]
                    if dtype == 'complex':
                        result[name] = complex(values[name])
                    else:
                        val = np.fromstring(values[name].strip('[]'), sep=',')
                        val = val.astype(dtype)
                        if shape:
                            val.resize(shape)
                        result[name] = val
                else:
                    result[name] = values[name]
            return result
        func.__name__ = fname
        task.func = func
        return task

    def get_current_result(self, name, squeeze=[]):
        if squeeze:
            squeeze = list(squeeze)[::-1]
            names = tuple(self.params._state_names)
            idxs = [names.index(name) for name in squeeze]
            levels = [self._levels[name] for name in squeeze]
            states = list(self.params._states)
            results = []
            for level in itertools.product(*levels):
                for n, idx in enumerate(idxs):
                    states[idx] = level[n]
                args, values = self._results[tuple(states)]
                results.append(values[name])
            return results
        else:
            args, values = self._results[tuple(self.params._states)]
            return values[name]

    @property
    def results(self):
        return self._get_results()

    @property
    def results_all(self):
        return self._get_results(with_underscores=True)

    def _get_results(self, with_underscores=False):
        datas = []
        names = []
        for states, (args, out) in self._results.items():
            if not names:
                names.extend('idx_' + '_'.join(n) for n in self.params._state_names)
                names.extend(args.keys())
                try:
                    for name in out.keys():
                        if not name.startswith('_'):
                            names.append(name + '_out')
                    for name in out.keys():
                        if name.startswith('_') and with_underscores:
                            names.append(name + '_out')
                except AttributeError:
                    names.append('out')
            line = list(states) + list(args.values())
            try:
                for name, value in out.items():
                    if not name.startswith('_'):
                        line.append(value)
                for name, value in out.items():
                    if name.startswith('_') and with_underscores:
                        line.append(value)
            except AttributeError:
                line.append(out)
            datas.append(line)
        idx_names = [name for name in names if name.startswith('idx_')]
        if datas:
            return pd.DataFrame(datas, columns=names).set_index(idx_names)
        else:
            return pd.DataFrame()

    def to_array(self):
        dtypes = []
        for n, (states, (args, out)) in enumerate(self._results.items()):
            if not dtypes:
                dtypes = [ ('state', 'O') ]
                for name, value in args.items():
                    dtype = (name + '_arg', type(value))
                    dtypes.append(dtype)
                for name, value in out.items():
                    if isinstance(value, np.ndarray):
                        dtype = (name + '_value', value.dtype, value.shape)
                    elif isinstance(value, (int, float, complex)):
                        dtype = (name + '_value', type(value))
                    else:
                        dtype = (name + '_value', 'O')
                    dtypes.append(dtype)
                datas = np.zeros(len(self._results), dtypes)
            data = (states,) + tuple(args.values()) + tuple(out.values())
            datas[n] = data
        return datas

    def join(self, other):
        @Task
        def joined(**kwargs):
            result = self.call()
            result.update(other.call())
            return result
        joined.func.__name__ = '{}_{}'.format(self.func.__name__,
            other.func.__name__)
        for name in self.params._params:
            joined.params._append(name, getattr(self.params, name))
        for name in other.params._params:
            joined.params._append(name, getattr(other.params, name))
        return joined

    @classmethod
    def from_file(cls, fname):
        task = read_json(open(fname + '.json').read())
        data = np.load(fname + '.npy')
        states = {state:n for n, state in enumerate(data['state'])}
        indices = {}
        for n, name in enumerate(data.dtype.names):
            if name.endswith('_value'):
                indices[n] = name.rsplit('_value')[0]
        def func(*args, **kwargs):
            idx =states[tuple(task.params._states)]
            values = data[idx]
            return {name:values[n] for n, name in indices.items()}
        func.__name__ = fname
        task.func = func
        return task

    @property
    def state(self):
        return tuple(self.params._states)

    def next_state(self):
        if self.is_running():
            sweeps = list(self.params._sweeps)
            idx = 0
            while sweeps[idx].is_finished():
                idx += 1
            sweeps[idx].next_state()
            while idx > 0:
                idx -= 1
                sweeps[idx].reset()

    def is_running(self):
        return any(sweep.is_running() for sweep in self.params._sweeps)

    def is_finished(self):
        return not self.is_running()

    def is_interrupting(self):
        for param, idx in self._interrupts.items():
            if param.value.idx in idx:
                sweeps = tuple(self.params._sweeps)
                inner_sweeps = sweeps[:sweeps.index(param.value)]
                for sweep in inner_sweeps:
                    if sweep.is_running():
                        return False
                return True
        return False

    def out(self):
        self.is_iterating = True
        return self.call()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        res = 1
        for sweep in self.params._sweeps:
            res *= len(sweep)
        return res


class ResultSweep(object):
    def __init__(self, task, names='', sweeps=[], squeeze=[]):
        self.task = task
        self.names = names.split() if isinstance(names, str) else names
        if sweeps == '*':
            self.sweeps = list(task.params._sweep_names)
        else:
            if isinstance(sweeps, str):
                sweeps = sweeps.split()
            self.sweeps = []
            for _sweep in sweeps:
                _name = ''
                for _names in self.task.params._sweep_names:
                    if _sweep in _names:
                        _name = '_'.join(_names)
                        break
                if _name:
                    self.sweeps.append(_names)
                else:
                    msg = ('Sweep name {!r} not found in sweeps '
                           'of task {!r}!').format(_sweep, self.task)
                    raise ValueError(msg)
        self.squeeze = squeeze
        if squeeze != '*':
            if isinstance(squeeze, str):
                squeeze = squeeze.split()
            self.squeeze = []
            self.squeezed_params = []
            for _sweep in squeeze:
                _name = ''
                for _names in self.task.params._sweep_names:
                    if _sweep in _names:
                        _name = '_'.join(_names)
                        self.squeezed_params.extend(_names)
                        break
                if _name:
                    self.squeeze.append(_name)
                else:
                    msg = ('Squeezing sweep name {!r} not found in sweeps '
                           'of task {!r}!').format(_sweep, self.task)
                    raise ValueError(msg)
        self._states = None
        self.is_iterating = False

    def __repr__(self):
        args = []
        args.append(repr(self.task))
        for name in ['names', 'sweeps', 'squeeze']:
            value = getattr(self, name)
            if value:
                arg = '{}={!r}'.format(name, value)
                args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


    @property
    def states(self):
        if self._states is None:
            _states = []
            for _names in self.sweeps:
                sweep = Iterate(*self.task._levels['_'.join(_names)])
                _states.append(sweep)
            self._states = _states
        return self._states

    def reset(self):
        self.is_iterating = False
        for sweep in self.states:
            sweep.reset()

    @property
    def min(self):
        return None
        return min(self.start, self.stop)

    @property
    def max(self):
        return None
        return max(self.start, self.stop)

    @property
    def state(self):
        state = list(self.task.params._states)
        state_names = tuple(self.task.params._state_names)
        for names, sweep in zip(self.sweeps, self.states):
            idx = state_names.index(names)
            state[idx] = sweep.out()
        return tuple(state)

    def out(self):
        self.is_iterating = True
        state_names = tuple(self.task.params._state_names)
        state = list(self.state)
        if self.squeeze == '*':
            results = {}
            names = []
            for args, values in self.task._results.values():
                if not names:
                    if self.names:
                        names = self.names
                    else:
                        names = ['arg_' + k for k in args.keys()]
                        names.extend(values.keys())
                for name in names:
                    vals = results.setdefault(name, [])
                    if name.startswith('arg_'):
                        vals.append(args[name.strip('arg_')])
                    else:
                        vals.append(values[name])
            return {k: np.asarray(v) for k, v in results.items()}
        elif self.squeeze:
            squeeze = list(self.squeeze)[::-1]
            idxs = [state_names.index(name.split('_')) for name in squeeze]
            levels = [self.task._levels[name] for name in squeeze]
            results = {}
            names = []
            const_arg_names = []
            for level in itertools.product(*levels):
                for n, idx in enumerate(idxs):
                    state[idx] = level[n]
                args, values = self.task._results[tuple(state)]
                if not names:
                    if self.names:
                        names = self.names
                    else:
                        names = ['arg_' + k for k in args.keys()]
                        names.extend(values.keys())
                for name in names:
                    vals = results.setdefault(name, [])
                    if name.startswith('arg_'):
                        arg_name = name.strip('arg_')
                        if arg_name in self.squeezed_params:
                            vals.append(args[arg_name])
                    else:
                        vals.append(values[name])
            results = {k: np.asarray(v) for k, v in results.items()}
            for name in names:
                if name.startswith('arg_'):
                    arg_name = name.strip('arg_')
                    if arg_name not in self.squeezed_params:
                        results[name] = args[arg_name]
            return results
        else:
            args, values = self.task._results[tuple(state)]
            if self.names:
                results = {}
                for name in self.names:
                    if name.startswith('arg_'):
                        results[name] = args[name.strip('arg_')]
                    else:
                        results[name] = values[name]
                return results
            else:
                results = {'arg_' + k: v for k, v in args.items()}
                results.update(values)
                return results

    def next_state(self):
        if self.is_running():
            sweeps = self.states
            idx = 0
            while sweeps[idx].is_finished():
                idx += 1
            sweeps[idx].next_state()
            while idx > 0:
                idx -= 1
                sweeps[idx].reset()

    def is_running(self):
        return any(sweep.is_running() for sweep in self.states)

    def is_finished(self):
        return not self.is_running()

    def __iter__(self):
        while self.is_running():
            if self.is_iterating:
                self.next_state()
            yield self.out()

    def as_list(self):
        return list(self)

    def __len__(self):
        res = 1
        for sweep in self.states:
            res *= len(sweep)
        return res

    @property
    def idx(self):
        sweeps = self.states
        if not sweeps:
            return 0
        idx = sweeps[0].idx
        for n in range(1, len(sweeps)):
            idx += sweeps[n].idx * len(sweeps[n-1])
        return idx


class Tasks(Iterate):
    def __init__(self, *tasks):
        Iterate.__init__(self, *tasks)
        self._interrupted_tasks = []

    def next_state(self):
        if self.idx < self._state_stop:
            self.idx = self.idx + 1
        else:
            self.idx = self._interrupted_tasks.pop()
        return self.idx

    def is_running(self):
        return self.idx < self._state_stop or \
               any(task.is_running() for task in self.items)


class TaskSweep(object):
    def __init__(self, *tasks):
        self.params = Parameters(self)
        self.tasks = Tasks(*tasks)
        self.actions = []
        self.post_actions = {}
        self._log_output = None
        self._plot_size = (6.4, 4.8)
        #~ self.reset()
        self._plot_update = True
        self.is_configured = False
        self._am = AxesManager()

    def configure(self):
        for task in self.tasks.items:
            for func, obj, args, kwargs in task._config:
                if func.__name__ in ('plot', 'linesweep'):
                    sig = inspect.getargspec(func)
                    func_args = dict(zip_longest(sig.args[::-1],
                                     sig.defaults[::-1]))
                    for name, val in zip(sig.args, args):
                        func_args[name] = val
                    func_args.update(kwargs)
                    row = func_args['row']
                    col = func_args['col']
                    self._am.append_loc(row, col)
                    task.params._pm.am = self._am

    def __repr__(self):
        args = []
        for item in self.tasks.items:
            arg = '{}'.format(repr(item))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['name'] = 'tasklist'
        params = self.params
        dct['params'] = [getattr(params, name) for name in params._params]
        dct['tasks'] = self.tasks.items
        return dct

    @classmethod
    def _from_dict(cls, dct):
        tasklist = cls(*dct['tasks'])
        for param in dct['params']:
            tasklist.params._append(param)
        return tasklist

    def to_json(self):
        return write_json(self)

    @classmethod
    def from_json(cls, s):
        return read_json(s)

    def add_param(self, name, value=None):
        self.params._append(Param(name, value))

    @property
    def task(self):
        return self.tasks.item

    @property
    def args(self):
        return self.task.args

    @property
    def sweeps(self):
        task_sweeps = tuple(self.task.params._sweeps)
        return task_sweeps + (self.tasks,) + tuple(self.params._sweeps)

    @property
    def state(self):
        state = tuple(self.task.params._states) + (self.tasks.state,)
        return state + tuple(self.params._states)

    @property
    def named_states(self):
        names = tuple(self.task.params._sweep_names) + ('task',)
        names += tuple(self.params._sweep_names)
        States = namedtuple('States', names)
        return States(*self.state)

    @property
    def named_sweeps(self):
        names = tuple(self.task.params._sweep_names) + ('task',)
        names += tuple(self.params._sweep_names)
        Sweeps = namedtuple('Sweeps', names)
        return Sweeps(*self.sweeps)

    def next_state(self):
        sweeps = self.sweeps
        idx = 0
        while sweeps[idx].is_finished():
            idx += 1
        sweeps[idx].next_state()
        idx = idx - len(sweeps)
        sweeps = self.sweeps    # refresh sweep list due to possible new task
        while idx > -len(sweeps):
            idx -= 1
            sweep = sweeps[idx]
            if sweep.is_finished():
                sweep.reset()
                sweeps = self.sweeps    # refresh sweep list
            elif sweep.is_iterating:
                sweep.next_state()
                sweeps = self.sweeps    # refresh sweep list

    def is_running(self):
        return any(sweep.is_running() for sweep in self.sweeps)

    def is_finished(self):
        return not self.is_running()

    def reset(self):
        for sweep in self.sweeps:
            sweep.reset()
        for task in self.tasks.items:
            task.params._pm.reset()
        try:
            self.fig.clf()
        except Exception:
            pass

    def action(self):
        for action in self.actions:
            action()

    def post_action(self):
        for idx in self.post_actions:
            if all(sweep.is_finished() for sweep in self.sweeps[:idx+1]):
                for action in self.post_actions[idx]:
                    action()

    def log(self, msg):
        if self._log_output:
            import IPython.display
            with self._log_output:
                IPython.display.clear_output()
                print(msg)
        else:
            print(msg)

    @property
    def out(self):
        if not self.is_configured:
            self.configure()
            self.is_configured = True

        name = self.task.func.__name__
        results = OrderedDict(state=self.state)
        results['func'] = self.task.func.__name__
        for name, param in reversed(list(self.params)):
            try:
                results[name] = param.out()
            except AttributeError:
                results[name] = param
        results['args'] = self.task.args
        output = self.task.call()
        results['out'] = output
        Out = namedtuple('Out', results.keys())
        for sweep in self.sweeps:
            sweep.is_iterating = True

        # logging
        sweep_params = []
        for name, param in self.params:
            if hasattr(param.value, 'state'):
                sweep = param.value
                num = sweep.idx + 1
                nums = len(sweep)
                value = param.out()
                param_str = '{}={} ({}/{})'.format(name, value, num, nums)
                sweep_params.append(param_str)
        num = self.tasks.idx + 1
        nums = len(self.tasks.items)
        name = self.task.func.__name__
        sweep_params.append('task {}/{}:  {}'.format(num, nums, name))
        msg = '    '.join(sweep_params)

        sweep_args = []
        depend_args = []
        const_args = []
        for name, param in self.task.params:
            if hasattr(param.value, 'state'):
                sweep = param.value
                num = sweep.idx + 1
                nums = len(sweep)
                value = param.out()
                param_str = '{}={} ({}/{})'.format(name, value, num, nums)
                if hasattr(param.value, 'next_state'):
                    sweep_args.append(param_str)
                else:
                    depend_args.append(param_str)
            else:
                param_str = '{}={}'.format(name, param.out())
                const_args.append(param_str)

        if sweep_args:
            msg += '\n    sweep args:  '
            msg += ',  '.join(sweep_args[::-1])
        if depend_args:
            msg += '\n   depend args:  '
            msg += ',  '.join(depend_args[::-1])
        if const_args:
            msg += '\n    const args:  '
            msg += ',  '.join(const_args)
        if output is not None:
            msg += '\n        output:  '
            try:
                out_str = []
                out_str2 = []
                for name, value in output.items():
                    if not name.startswith('_'):
                        out_str.append('{}={}'.format(name, value))
                    else:
                        _fmt = '\n                {}=...'
                        out_str2.append(_fmt.format(name))
                msg += ',  '.join(out_str)
                msg += ''.join(out_str2)
            except AttributeError:
                msg += '{}'.format(output)
        self.log(msg)

        # plotting
        if self.task.params._pm.plots:
            _states = tuple(self.task.params._states)
            _args = self.task.params._args
            self.task.params._pm.plot(_states, _args, output)
            if self._plot_update:
                self._am.update()
        return Out(**results)
        #~ result = self.task.call()
        #~ self.pm.plot(self.state, result)
        #~ self.pm.update()
        #~ self.action()
        #~ self.post_action()

    def gui(self, plot_size=(9, 4)):
        self._plot_size = plot_size
        return gui.gui(self)

    @property
    def is_iterating(self):
        return all(s.is_iterating for s in self.sweeps)

    def next_out(self):
        if self.task.is_interrupting():
            self.tasks._interrupted_tasks.append(self.tasks.state)
            self.tasks.state = self.tasks.next_state()
        if self.is_running() and self.is_iterating:
            self.next_state()
        return self.out

    def run(self, idx=None, inner=False, plot_update=True):
        """run the sweeps of the tasklist

        tasklist.run()           run until tasklist is finished

        idx >= 0        regards the sweeps of current task
        idx <  0        regards the tasklist sweeps (tasks and params)

        tasklist.run(0)          tasklist.next_out()
        tasklist.run(1)          next state of second sweep until
                                 inner states are finished
        tasklist.run(1, True)    next state of second sweep until
                                 the same inner state
        """
        self._plot_update = plot_update
        inner_pre = self.state[:idx]
        while self.is_running():
            self.next_out()
            if all([sweep.is_finished() for sweep in self.sweeps[:idx]]):
                break
        tidx = self.sweeps.index(self.tasks) - len(self.state)
        if inner and (self.task.params._is_running() or tidx < idx < 0):
            while self.is_running() and self.state[:idx] != inner_pre:
                self.next_out()

    def __iter__(self):
        self.reset()
        while self.is_running():
            yield self.next_out()

    def as_list(self):
        return list(self)


class SweepPlot(object):
    def __init__(self, param_name, idx, x='', y='', use_cursor=True,
                 xlim=(None, None), xlabel='', ylabel='',
                 sorted_xdata=False, slices={}, **kwargs):
        self.param_name = param_name
        self.idx = idx
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel if ylabel else y
        self.xlim = xlim
        self.use_cursor = use_cursor
        self.sorted_xdata = sorted_xdata
        self.slices = slices
        self.kwargs = kwargs
        self.reset()

    def reset(self):
        self.skeys = {}
        self.keys = {}
        self.lines = {}
        self.cursor = None
        self.annotation = None

    def __repr__(self):
        args = []
        for name in 'idx', 'x', 'y', 'xlabel', 'ylabel', 'xlim':
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        if not self.use_cursor:
            arg = 'use_cursor=False'
            args.append(arg)
        if self.sorted_xdata:
            arg = 'sorted_xdata=True'
            args.append(arg)
        for name, value in self.kwargs.items():
            arg = '{}={}'.format(name, repr(value))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['idx'] = self.idx
        dct['x'] = self.x
        dct['y'] = self.y
        dct['xlabel'] = self.xlabel
        dct['ylabel'] = self.ylabel
        dct['xlim'] = self.xlim
        dct['use_cursor'] = self.use_cursor
        dct['sorted_xdata'] = self.sorted_xdata
        dct['kwargs'] = self.kwargs
        return dct

    @classmethod
    def _from_dict(cls, dct):
        kwargs = dct.pop('kwargs')
        dct.update(kwargs)
        return cls(**dct)

    def _max(self, val):
        if val is None:
            return -1

    def _pop(self, idxs, items):
        idx = idxs[0]
        ipos = idx if idx >= 0 else len(items) + idx
        if len(idxs) > 1:
            return items[:ipos] + (self._pop(idxs[1:], items[ipos]),) + items[ipos +1:]
        else:
            return items[:ipos] + items[ipos+1:]

    def plot(self, ax, key, args, output):
        for pos, idxs in self.slices.items():
            if key[pos] not in idxs:
                return
        data = output
        skey = self._pop(self.idx, key)
        xval = data[self.x] if self.x else args[self.param_name]
        yval = data[self.y] if self.y else data
        yval = np.nan if yval is None else yval
        try:
            line = self.skeys[skey]
            data = self.lines[line]
            if self.sorted_xdata:
                data[xval] = yval
                xdata = data.keys()
                ydata = data.values()
            else:
                xdata, ydata = data
                xdata.append(xval)
                ydata.append(yval)
            line.set_xdata(xdata)
            line.set_ydata(ydata)

            #~ xdata = line.get_xdata()
            #~ xdata.append(xval)

            #~ #xdata = np.append(xdata, xval)
            #~ line.set_xdata(xdata)

            #~ ydata = line.get_ydata()
            #~ ydata.append(yval)
            #~ #ydata = np.append(ydata, yval)
            #~ line.set_ydata(ydata)

            self.keys[key] = line, len(xdata)-1

            #~ self.lines[line].append(self.out())
        except KeyError:
            kwargs = dict(self.kwargs)
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
            line, = ax.plot(xval, yval, picker=10, **kwargs)
            if self.sorted_xdata:
                data = SortedDict({xval: yval})
                xdata = data.keys()
                ydata = data.values()
            else:
                xdata = [xval]
                ydata = [yval]
                data = xdata, ydata
            line.set_xdata(xdata)
            line.set_ydata(ydata)

            #~ line.set_xdata(list(line.get_xdata()))
            #~ line.set_ydata(list(line.get_ydata()))
            self.skeys[skey] = line
            self.lines[line] = data
            self.keys[key] = line, len(line.get_xdata())-1
        self.set_cursor(ax, xval, yval, line.get_color())
        return line

    def update_cursor(self, ax, key, args, output, **kwargs):
        line, idx = self.keys[key]
        x = line.get_xdata()[idx]
        y = line.get_ydata()[idx]
        self.set_cursor(ax, x, y, line.get_color())
        return y

    def set_cursor(self, ax, xval, yval, color):
        if self.use_cursor:
            if self.cursor is None:
                self.cursor, = ax.plot(np.nan, np.nan,
                    marker='o',
                    markersize=12,
                    alpha=0.5)
            self.cursor.set_data([xval, yval])
            #~ self.cursor.set_color(shade_color(line.get_color(), 50))
            self.cursor.set_color(color)
            if self.annotation is None:
                self.annotation = ax.annotate(
                    s='anno',
                    xy=(np.nan, np.nan),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=11 ,
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        alpha=0.9,
                        edgecolor='none',
                    ),
                    visible=False,
                )
            self.annotation.xy = xval, yval
            bbox = self.annotation.get_bbox_patch()
            bbox.set_facecolor(self._shade_color(color, 50))
            self.annotation.set_visible(False)

    def draw(self, ax):
        if self.cursor is not None:
            ax.add_line(self.cursor)
        if self.annotation is not None:
            ax._add_text(self.annotation)
        for line in self.lines:
            ax.add_line(line)

    @staticmethod
    def _shade_color(color, percent):
        """ A color helper utility to either darken or lighten given color

        from https://github.com/matplotlib/matplotlib/pull/2745
        """
        rgb = colorConverter.to_rgb(color)
        h, l, s = rgb_to_hls(*rgb)
        l *= 1 + float(percent)/100
        l = np.clip(l, 0, 1)
        r, g, b = hls_to_rgb(h, l, s)
        return r, g, b


class LinePlot(object):
    def __init__(self, x='', y='', xlabel=None, ylabel='',
                       dx='', xoffs='', **kwargs):
        self.x = x
        self.y = y
        self.dx = dx
        self.xoffs = xoffs
        self.xlabel = x if xlabel is None else xlabel
        self.ylabel = ylabel
        self.kwargs = kwargs
        self.line = None
        self.keys = {}
        self.label = kwargs.pop('label', self.y)

    def __repr__(self):
        args = []
        for name in 'x', 'y':
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        for name, value in self.kwargs.items():
            arg = '{}={}'.format(name, repr(value))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _to_dict(self):
        dct = OrderedDict(__class__=self.__class__.__name__)
        dct['x'] = self.x
        dct['y'] = self.y
        dct['kwargs'] = self.kwargs
        return dct

    @classmethod
    def _from_dict(cls, dct):
        return cls(x=dct['x'], y=dct['y'], **dct['kwargs'])

    def plot(self, ax, key, args, output):
        data = output
        if self.line is not None:
            if self.y:
                ydata = data[self.y]
                if self.x:
                    self.line.set_xdata(data[self.x])
                    self.line.set_ydata(ydata)
                elif isinstance(ydata, (int, float)):
                    self.line.set_ydata([ydata, ydata])
                elif self.dx:
                    dx = data[self.dx]
                    offs = data[self.xoffs] if self.xoffs else 0
                    xdata = np.arange(len(ydata)) * dx + offs
                    self.line.set_xdata(xdata)
                    self.line.set_ydata(ydata)
                else:
                    self.line.set_ydata(data[self.y])
            else:
                xdata = data[self.x]
                if isinstance(xdata, (int, float)):
                    self.line.set_xdata([xdata, xdata])
        else:
            if self.y:
                ydata = data[self.y]
                if self.x:
                    xdata = data[self.x]
                    self.line, = ax.plot(xdata, ydata, **self.kwargs)
                elif isinstance(ydata, (int, float)):
                    self.line = ax.axhline(ydata, **self.kwargs)
                elif self.dx:
                    dx = data[self.dx]
                    offs = data[self.xoffs] if self.xoffs else 0
                    xdata = np.arange(len(ydata)) * dx + offs
                    self.line, = ax.plot(xdata, ydata, **self.kwargs)
                else:
                    self.line, = ax.plot(ydata, **self.kwargs)
            else:
                xdata = data[self.x]
                if isinstance(xdata, (int, float)):
                    self.line = ax.axvline(xdata, **self.kwargs)
                    if self.label == self.y:
                        self.label = self.x
        return self.line

    def update_cursor(self, ax, key, args, output, **kwargs):
        self.plot(ax, key, args, output, **kwargs)

    def reset(self):
        self.line = None

    def draw(self, ax):
        ax.add_line(self.line)


class SweepLinePlot(LinePlot):
    def __init__(self, x='', y='', use_cursor=True, **kwargs):
        LinePlot.__init__(self, x, y, **kwargs)
        self.cursor = None
        self.use_cursor = use_cursor

    def plot(self, ax, key, args, output):
        data = output
        ydata = data[self.y]
        xdata = data[self.x]
        if key in self.keys:
            line = self.keys[key]
            line.set_data([xdata, ydata])
            self.set_cursor(ax, xdata, ydata, line.get_color())
        else:
            line, = ax.plot(xdata, ydata, **self.kwargs)
            self.keys[key] = line
            self.set_cursor(ax, xdata, ydata, line.get_color())
        return line

    def update_cursor(self, ax, key, args, output, **kwargs):
        line = self.keys[key]
        xdata, ydata = line.get_data()
        self.set_cursor(ax, xdata, ydata, line.get_color())

    def set_cursor(self, ax, xdata, ydata, color):
        if self.use_cursor:
            if self.cursor is None:
                self.cursor, = ax.plot(np.nan, np.nan,
                    linewidth=7, alpha=0.5, zorder=1)
            self.cursor.set_data([xdata, ydata])
            #~ self.cursor.set_color(shade_color(line.get_color(), 50))
            self.cursor.set_color(color)


class PlotManager(object):
    def __init__(self, task):
        self.plots = OrderedDict()
        self.task = task
        self.reset()

    def reset(self):
        for plot in self.plots:
            plot.reset()
        self.lines = {}
        self.leglines = {}
        self.is_configured = False
        self.am = None

    def configure(self):
        if not self.is_configured and self.plots:
            idxs = {}
            for plot, (row, col) in self.plots.items():
                xax = None
                if isinstance(plot, SweepPlot) and plot.idx in idxs:
                    xax = idxs[plot.idx]
                ax = self.am.get_axes(row, col, xax)
                self.plots[plot] = ax
                ax.grid(True)
                if plot.xlabel:
                    ax.set_xlabel(plot.xlabel)
                if plot.ylabel:
                    ax.set_ylabel(plot.ylabel,
                        rotation='horizontal', ha='right', labelpad=10)
                if isinstance(plot, SweepPlot):
                    left, right = plot.xlim
                    if None not in (left, right):
                        offs = 0.025 * abs(right - left)
                        ax.set_xlim(left-offs, right+offs)
                    if plot.idx not in idxs:
                        idxs[plot.idx] = ax
            self.am.fig.tight_layout()
            self.is_configured = True

    def add_sweep_plot(self, param_name, idx, x='', y='', row=0, col=0,
                       xlim=None, xlabel='', ylabel='', use_cursor=True,
                       slices={}, **kwargs):
        plot = SweepPlot(param_name, idx, x, y,
            use_cursor=use_cursor,
            xlim=xlim,
            xlabel=xlabel,
            ylabel=ylabel,
            slices=slices,
            **kwargs)
        self.plots[plot] = row, col

    def add_line_plot(self, x='', y='', row=0, col=0,
                      xlabel=None, ylabel='', dx='', xoffs='', **kwargs):
        plot = LinePlot(x, y, xlabel, ylabel, dx=dx, xoffs=xoffs, **kwargs)
        self.plots[plot] = row, col

    def add_linesweep_plot(self, param_name, idx, x='', y='', row=0, col=0,
                       xlim=None, xlabel='', ylabel='', use_cursor=True,
                       slices={}, **kwargs):
        plot = SweepLinePlot(x, y, **kwargs)
        self.plots[plot] = row, col

    def plot(self, key, args, output, **kwargs):
        leg_axs = {}
        for plot, ax in self.plots.items():
            line = plot.plot(ax, key, args, output, **kwargs)
            keys = self.lines.setdefault(line, [])
            keys.append(key)
            self.am.lines[line] = self
            if hasattr(plot, 'label') and plot.label:
                lines = leg_axs.setdefault(ax, [])
                lines.append((line, plot.label))
        # create legend
        self.leglines = {}
        for ax, legs in leg_axs.items():
            lines, labels = zip(*legs)
            leg = ax.legend(lines, labels)
            # make legend clickable
            for line, legline in zip(lines, leg.get_lines()):
                legline.set_picker(10)
                self.leglines[legline] = line
                self.am.lines[legline] = self


class AxesManager(object):
    def __init__(self):
        self.rows = 1
        self.cols = 1
        self.locs = {}
        self.fig = None
        self.lines = {}

    @staticmethod
    def _loc(loc):
        if isinstance(loc, (tuple, list)):
            if loc[1] is not None:
                return (loc[0], loc[1]+1)
            else:
                return (loc[0], loc[1])
        elif loc is None:
            return (None, None)
        else:
            return (loc, loc+1)

    def append_loc(self, row, col):
        ridx = self._loc(row)
        cidx = self._loc(col)
        self.rows = max(self.rows, 1 if ridx[1] is None else ridx[1])
        self.cols = max(self.cols, 1 if cidx[1] is None else cidx[1])
        self.locs[row, col] = (ridx, cidx), None
        return ridx, cidx

    def get_axes(self, row, col, xax=None):
        (ridx, cidx), ax = self.locs[row, col]
        if ax is None:
            gs = GridSpec(self.rows, self.cols)
            if self.fig is None:
                self.fig = plt.figure()
            self.add_subplot_zoom(self.fig)
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
            ax = self.fig.add_subplot(gs[slice(*ridx), slice(*cidx)], sharex=xax)
            self.locs[row, col] = (ridx, cidx), ax
            return ax
        else:
            return ax

    def update(self):
        for ax in self.fig.axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_pick(self, event):
        line = event.artist
        idx = event.ind[0]
        pm = self.lines[line]
        if line in pm.leglines:
            plotted_line = pm.leglines[line]
            vis = not plotted_line.get_visible()
            plotted_line.set_visible(vis)
            line.set_alpha(1.0 if vis else 0.2)
            line.figure.canvas.draw()
            line.figure.canvas.flush_events()
        else:
            keys = pm.lines[line]
            key = keys[idx]
            sweeped_args = list(pm.task.params._sweep_values)
            args, output = pm.task._results[key]
            for plot, ax in pm.plots.items():
                y = plot.update_cursor(ax, key, sweeped_args, output)
                if hasattr(plot, 'annotation'):
                    state_args = OrderedDict()
                    for names in pm.task.params._state_names:
                        for name in names:
                            state_args[name] = args[name]
                    if y is None:
                        anno_text = []
                    else:
                        anno_text = ['{}: {:.7g}'.format(plot.y, y)]
                    for n, v in state_args.items():
                        try:
                            anno_text.append('{} = {:.6g}'.format(n, v))
                        except (ValueError, TypeError):
                            if not isinstance(v, (np.ndarray, list, tuple)):
                                anno_text.append('{} = {}'.format(n, v))
                    plot.annotation.set_text('\n'.join(anno_text))
                    plot.annotation.set_visible(True)
            self.update()

    def add_subplot_zoom(self, fig):
        # from
        # https://www.semipol.de/2015/09/04/matplotlib-interactively-zooming-to-a-subplot.html
        # temporary store for the currently zoomed axes. Use a list to work around
        # python's scoping rules
        zoomed_axes = [None]

        def on_zoom_click(event):
            ax = event.inaxes
            if ax is None:
                # occurs when a region not in an axis is clicked...
                return
            # we want to allow other navigation modes as well. Only act in case
            # shift was pressed and the correct mouse button was used
            if event.key != 'shift' or event.button != 1:
                return
            if zoomed_axes[0] is None:
                # not zoomed so far. Perform zoom
                # store the original position of the axes
                zoomed_axes[0] = (ax, ax.get_position())
                ax.set_position([0.1, 0.1, 0.85, 0.85])
                # hide all the other axes...
                for axis in event.canvas.figure.axes:
                    if axis is not ax:
                        axis.set_visible(False)
            else:
                # restore the original state
                zoomed_axes[0][0].set_position(zoomed_axes[0][1])
                zoomed_axes[0] = None
                # make other axes visible again
                for axis in event.canvas.figure.axes:
                    axis.set_visible(True)
            # redraw to make changes visible.
            event.canvas.draw()
            event.canvas.flush_events()
        fig.canvas.mpl_connect('button_press_event', on_zoom_click)


class TabWriter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.indents = [0]
        self.s = ''
        self.llen = 0

    def write(self, s):
        self.s += s
        self.llen += len(s)
        #~ self.indents[-1] += len(s)

    def newline(self):
        self.s += '\n' + ' ' * self.indents[-1]
        self.llen = 0

    def tab(self):
        self.indents.append(self.indents[-1] + self.llen)
        self.llen = 0

    def untab(self):
        self.indents.pop()

    def to_json(self, o):
        """code from ...
        https://stackoverflow.com/questions/10097477/python-json-array-newlines
        """
        if isinstance(o, dict):
            self.write("{")
            self.tab()
            for n, (k, v) in enumerate(o.items()):
                self.write('"' + str(k) + '": ')
                self.to_json(v)
                if n < len(o) - 1:
                    self.write(',')
                    self.newline()
            self.write("}")
            self.untab()
        elif isinstance(o, str):
            self.write('"' + o + '"')
        elif isinstance(o, (list, tuple)):
            self.write("[")
            self.tab()
            for n, e in enumerate(o):
                self.to_json(e)
                if n < len(o) - 1:
                    self.write(',')
                    if isinstance(e, dict) or hasattr(e, '_to_dict'):
                        self.newline()
                    else:
                        self.write(' ')
            self.write("]")
            self.untab()
        elif isinstance(o, bool):
            self.write("true" if o else "false")
        elif isinstance(o, (int, float)):
            self.write('{!r}'.format(o))
        elif isinstance(o, np.ndarray):
            self.to_json(o.flatten().tolist())
        elif o is None:
            self.write('null')
        else:
            self.to_json(o._to_dict())
        return self.s


class Slice(object):
    def __init__(self, start=None, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step

    def __contains__(self, idx):
        is_start = (not self.start or self.start <= idx)
        is_stop = (not self.stop or idx <= self.stop)
        idx_start = idx if self.start is None else idx - self.start
        is_step = not (self.step and idx_start % self.step)
        return is_start and is_stop and is_step


if __name__ == '__main__':
    #~ def a():
        #~ return 2.3
    #~ b = dict(a=1, b=2)

    #~ d1 = Dependency(a)
    #~ assert d1.eval() == 2.3

    #~ d2 = Dependency(b, 'b')
    #~ b['b'] = 111
    #~ assert d2.eval() == 111

    #~ d3 = Dependency(b.__len__)
    #~ assert d3.eval() == len(b)


    from numpy import pi, sin

    @Task
    def myquad(m=1, x=0, b=0, c=[1, 2, 3]):
        y = m*x**2 + b
        z = 3+5j
        _x  = np.arange(10)*10
        _y  = np.random.randn(10) + b
        _y2 = np.random.randn(10)
        _z  = np.random.randn(2, 3).astype(complex)
        return y, z, _x, _y, _y2, _z

    @Task
    def mysin(t=0, freq=1, A=1):
        return A*sin(2*pi*freq * t)

    @Task
    def myid(x=0, m=1, b=0):
        return m*x + b

    @Task
    def pre(x=0):
        pass

    @Task
    def post(x=0):
        pass

    @Task
    def calc(a=None, b=None, m=None, y=None, x=None):
        pass

    #~ tasklist = TaskSweep(myid, myid, mysin, myquad, myid, myid)
    #~ tasklist = TaskSweep(myinit, myid, myquad, mysin, myid)
    #~ tasklist = TaskSweep(myinit, myid, myquad, myid)
    #~ tasklist = TaskSweep(myinit, myquad, myid)
    #~ tasklist = TaskSweep(pre, myquad, post)
    #~ tasklist = TaskSweep(pre, mysin, myquad, post)
    #~ tasklist = TaskSweep(pre, mysin, post)
    #~ tasklist = TaskSweep(mysin)
    tasklist = TaskSweep(myquad, calc)

    #~ tasklist.add_param('num', Iterate(10, 20))

    # myid
    myid.params.x.sweep(0, 2)
    myid.params.m.iterate(1, 2, 5)
    myid.params.m.interrupt(start=1)

    # mysin
    mysin.params.t.sweep(0, 1, num=5)
    mysin.params.t.sweep(1, 2, num=5)
    #~ mysin.params.A.iterate(1, 10)
    #~ mysin.params.freq.iterate(1, 0.5)
    mysin.params.t.plot(sub_index=1, row=4, col=None)

    # myquad
    myquad.params.x.sweep(0, 4)
    #~ myquad.params.x.iterate(3, 4)
    myquad.params.m.iterate(2, 1)
    #~ myquad.params.m.depends_on(tasklist.params.num)
    myquad.params.b.iterate(0, 5, zip='m')
    #~ myquad.params.b.iterate(0, 1, 2, 3)
    #~ myquad.params.b.iterate(0, 5)
    myquad.plot(y='_y')
    myquad.plot(y='y', ls=':', label=None)
    myquad.plot(x='y', ls='--', label=None)
    #~ myquad.params.x.plot(y='y', row=(1, 2), col=(1, 2), ylabel='y / mA',
        #~ slices=dict(b=[0, 3], x=Slice(step=2)))
    myquad.params.x.linesweep(x='_x', y='_y', row=1, col=(1, 2))
    myquad.params.x.plot(y='y', row=2, col=(1, 2))
    myquad.params.x.plot(y='y', row=0, col=(1, 2))
    myquad.params.m.plot(y='y', row=(1, 2), col=0)
    myquad.plot('_x', '_y', row=3, col=None, marker='o', lw=1.5)
    myquad.plot('_x', '_y2', row=3, col=None, ls=':')

    #~ calc.depends_on_task(myquad, sweeps='*', params=dict(
    #~ calc.depends_on_task(myquad, sweeps='b x', params=dict(
    calc.depends_on_task(myquad, squeeze='x', sweeps='b', params=dict(
    #~ calc.depends_on_task(myquad, squeeze='b', sweeps='x', params=dict(
        b='arg_b',
        m='arg_m',
        x='arg_x',
        y='y',
    ))

"""
# ToDo:

[p] select task to plot
    tasklist.select(myquad)
    tasklist.select(2)

    tp.tests.mytest.select()
    tp.select(tp.tests.mytest)
    tp.select(2)


[ ] choose sweep to plot
    up = myquad.params.x.sweep(0, 5, num=11)
    up.plot(y='y')

    down = myquad.params.x.sweep(0, 5, num=11, dir='down')
    down.plot(y='y')
    myquad.plot(y='y', sweep=down)

"""


