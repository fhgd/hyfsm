import collections
from collections import OrderedDict
from functools import wraps
import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import colorConverter
from colorsys import rgb_to_hls, hls_to_rgb

import task as tasksweep


class FSM(object):
    def __init__(self, state=None, graph=None, name='', trj_max=20, debug=False):
        self.state = state
        self.initial_state = state
        self.last_event = None
        self.last_state = None
        self.graph = collections.OrderedDict() if graph is None else graph
        self.actions = collections.OrderedDict()
        self.debug = debug
        if name:
            self.__name__ = name
        # add true event without a relation to this class (for naming)
        def true():
            return True
        self.true = true
        # save the last trj_max steps of the state trajectory
        self.trj = collections.deque(maxlen=trj_max)

    def init_state(self, state):
        """initialize data structure and FSM state"""
        if state not in self.graph:
            self.graph[state] = {}
        if state not in self.actions:
            try:
                self.actions[state] = [getattr(self, state)]
            except AttributeError:
                self.actions[state] = []
        # try to initialize FSM state
        if self.state is None:
            self.state = state
            self.initial_state = state

    def reset(self):
        self.state = self.initial_state

    def add_state(self, state, transitions=[], action=None):
        self.init_state(state)

        # add transitions
        for trn in list(transitions):
            event = trn[0]
            trn_actions = list(trn[1:-1])
            newstate = trn[-1]
            self.graph[state][event] = trn_actions, newstate
            self.init_state(newstate)

        # add state action(s)
        if action:
            if callable(action):
                self.actions[state].append(action)
            else:
                self.actions[state].extend(action)

    def add_transition(self, *args, **kwargs):
        """Add new transition from state if event is true to newstate.

        If event is omitted then true is assumed.
        """
        if len(args) == 2:
            state, newstate = args
            event = self.true
            action = kwargs.pop('action', None)
        elif len(args) == 3:
            state, event, newstate = args
            action = kwargs.pop('action', None)
        elif len(args) == 4:
            state, event, newstate, action = args

        self.init_state(state)
        self.init_state(newstate)

        actions = []
        if action:
            if callable(action):
                actions = [action]
            else:
                actions = list(action)
        self.graph[state][event] = actions, newstate

    def remove_transition(self, state, event):
        """Remove transition from state with event."""
        self.graph[state].pop(event)

    def add_action(self, action, state, event=None):
        actions = [action] if callable(action) else action
        if event:
            # append to transition
            self.graph[state][event][0].extend(actions)
        else:
            # append to state
            self.actions[state].extend(actions)

    @property
    def state_idx(self):
        return self.actions.keys().index(self.state)

    def get_active_event(self):
        """Return the first true event for the actual state."""
        transitions = self.graph[self.state]
        for event in transitions:
            if event():
                return event

    def next_state(self, *args, **kwargs):
        event = self.get_active_event()
        if event:
            # make state transition and action
            actions, newstate = self.graph[self.state][event]
            self.last_event = event
            self.last_state = self.state
            output = self.out()
            if output is None:
                self.trj.append(self.last_state)
            elif isinstance(output, (list, tuple)):
                self.trj.append((self.last_state,) + tuple(output))
            else:
                self.trj.append((self.last_state, output))

            if self.debug:
                logstr = '{} => {}'.format(
                    get_name(event), newstate)
                if hasattr(self, '__name__'):
                    logstr = '{}: '.format(self.__name__) + logstr
            for fun in actions:
                fun()
            self.state = newstate
            ret = [func() for func in self.actions[newstate]]
            if self.debug:
                logstr += ' {}'.format(self.out())
                #~ log.info(logstr)
                print(logstr)

            if len(ret) == 1:
                return ret[0]
            else:
                return ret

    def next_steady_state(self, *args, **kwargs):
        ret = self.next_state(*args, **kwargs)
        while self.get_active_event():
            ret = self.next_state()
        return ret

    def out(self):
        self.state

    def as_dot(self, ranks=None, max_size=None, rankdir='LR', nodesep=None):
        import pydot as dot
        import os

        if os.sys.platform == 'win32':
            dirs = 'Anaconda', 'Library', 'bin'
            for path in os.environ['PATH'].split(os.pathsep):
                if all(dir in path for dir in dirs):
                    graphviz = path + os.path.sep + 'graphviz'
                    os.environ['PATH'] += os.pathsep + graphviz
                    break

        properties = dict(
            colorscheme='greens7',
            color=5,
            fontcolor=7,
            fontsize=10,
            fontname='helvetica',
        )
        dot_kwargs = dict()
        if nodesep:
            g = dot.Dot(
                graph_type='digraph',
                rankdir=rankdir,
                newrank=True,
                nodesep=nodesep,
            )
        else:
            g = dot.Dot(
                graph_type='digraph',
                rankdir=rankdir,
                newrank=True,
            )
        g.set_node_defaults(
            shape='record',
            style='filled, rounded',
            fillcolor=1,
            **properties)
        g.set_edge_defaults(
            arrowsize=0.5,
            arrowhead="vee",
            **properties)

        if ranks:
            for name in ranks:
                nodes = ranks[name]
                if 'rank' in name:
                    s = dot.Subgraph(rank='same', graph_name=name)
                else:
                    s = dot.Subgraph(color='invis', graph_name=name)
                for node in nodes:
                    s.add_node(dot.Node(node))
                g.add_subgraph(s)

        fmtstate = '<FONT POINT-SIZE="12">{}</FONT>'
        fmtact = '<I>{}</I>'

        # newline
        fmtn = ' | '
        fmtn = '<br/>'

        graph = self.graph
        for state in graph:
            transitions = graph[state]
            for event in transitions:
                trnacts, newstate = transitions[event]
                acts = [(fmtn+fmtact).format(get_name(a)) for a in
                    self.actions[state] if not a.__name__.startswith('_')]
                label = '<{}>'.format(fmtstate.format(state) + ''.join(acts))
                node1 = dot.Node(state, label=label)
                g.add_node(node1)

                acts = [(fmtn+fmtact).format(get_name(a)) for a in
                    self.actions[newstate] if not a.__name__.startswith('_')]
                label = '<{}>'.format(fmtstate.format(newstate) + ''.join(acts))
                node2 = dot.Node(newstate, label=label)
                g.add_node(node2)

                trnacts = [('<br/>'+fmtact).format(get_name(a)) for a in
                    trnacts if not a.__name__.startswith('_')]
                trnstr = '<{}>'.format(get_name(event) + ''.join(trnacts))
                if (state == self.last_state) and (event == self.last_event) and (newstate == self.state):
                    e = dot.Edge(node1, node2, label=trnstr, penwidth=2)
                else:
                    e = dot.Edge(node1, node2, label=trnstr)
                g.add_edge(e)
        g.add_node(dot.Node(self.state, fillcolor=2, penwidth=2))
        if max_size:
            g.set_size(str(max_size))
        return g

    def draw(self, filename='', **kwargs):
        g = self.as_dot(**kwargs)
        if filename:
            if filename.endswith('.pdf'):
                g.write_pdf(filename)
            elif filename.endswith('.png'):
                g.write_png(filename)
            elif filename.endswith('.svg'):
                g.write_svg(filename)
        else:
            from IPython.display import SVG
            return SVG(g.create_svg())
            #~ return g.to_string()


def get_name(func):
    try:
        name = get_name(func.__self__)
        return '{}.{}'.format(name, func.__name__)
    except AttributeError:
        return func.__name__


class PlotManager(object):
    def __init__(self, task=None, hyfsm=None):
        self.plots = OrderedDict()
        self.reset()
        self.task = task
        self.hyfsm = hyfsm

    def reset(self):
        for plot in self.plots:
            plot.reset()
        self.lines = {}
        self.leglines = {}
        self.is_configured = False
        self.am = None
        self.plot_datas = {}
        self.states = (0,)

    def configure(self):
        if not self.is_configured and self.plots:
            idxs = {}
            state_plots = OrderedDict()
            for plot, (row, col) in self.plots.items():
                xax = None
                plt_types = tasksweep.SweepPlot, tasksweep.StatePlot
                if isinstance(plot, plt_types) and plot.idx in idxs:
                    xax = idxs[plot.idx]
                ax = self.am.get_axes(row, col, xax)
                self.plots[plot] = ax
                ax.grid(True)
                if plot.xlabel:
                    ax.set_xlabel(plot.xlabel)
                else:
                    # NEW
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.axes.xaxis.set_ticks_position('none')
                if plot.ylabel:
                    ax.set_ylabel(plot.ylabel,
                                  rotation='horizontal',
                                  ha='right',
                                  va='center',
                                  labelpad=10)
                if isinstance(plot, tasksweep.SweepPlot):
                    left, right = plot.xlim
                    if None not in (left, right):
                        offs = 0.025 * abs(right - left)
                        ax.set_xlim(left-offs, right+offs)
                    if plot.idx not in idxs:
                        idxs[plot.idx] = ax
                if isinstance(plot, tasksweep.StatePlot):
                    state_plots.setdefault(ax, []).append(plot)
                    ticks = []
                    labels = []
                    for p in state_plots[ax]:
                        ticks.append(p.yoffs)
                        labels.append(p.ylabel)
                    ax.set_ylabel('')
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(labels)
            self.am.fig.tight_layout()
            self.is_configured = True

    def add_sweep_plot(self, arg, row, col, xlabel, ylabel, use_cursor, **kwargs):
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        plot = tasksweep.SweepPlot(
            param_name=arg.name,
            idx=(0,),
            x='time',
            y='value',
            use_cursor=use_cursor,
            xlim=(None, None),
            xlabel='time / (clock ticks)' if xlabel is None else xlabel,
            ylabel=arg.name if ylabel is None else ylabel,
            **kwargs)
        self.plots[plot] = row, col
        self.plot_datas[plot] = arg

    def add_state_plot(self, arg, row, col, xlabel, ylabel, use_cursor,
                       **kwargs):
        plot = tasksweep.StatePlot(
            idx=(0,),
            x='time',
            y='value',
            use_cursor=use_cursor,
            xlabel='time / (clock ticks)' if xlabel is None else xlabel,
            ylabel=arg.name if ylabel is None else ylabel,
            **kwargs)
        for p, (r, c) in self.plots.items():
            if r == row and c == col:
                if isinstance(p, tasksweep.StatePlot):
                    plot.yoffs -= 1
        self.plots[plot] = row, col
        self.plot_datas[plot] = arg

    def plot(self):
        leg_axs = {}
        for plot, ax in self.plots.items():
            output = {}
            output['time'] = self.states[0]
            output['value'] = self.plot_datas[plot].value
            key = self.states
            line = plot.plot(ax, key, {}, output)
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
        self.states = (self.states[0] + 1,)


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
            for plot, ax in pm.plots.items():
                y = plot.update_cursor(ax, key, args=None, output=None)
                if hasattr(plot, 'annotation'):
                    state_args = OrderedDict()
                    #~ for names in pm.task.params._state_names:
                        #~ for name in names:
                            #~ state_args[name] = args[name]
                    if y is None:
                        anno_text = []
                    else:
                        #~ anno_text = ['{}: {:.7g}'.format(plot.y, y)]
                        anno_text = ['{:.7g}'.format(y)]
                    param = pm.plot_datas[plot]
                    hyfsm = param._parent
                    fsm_state = hyfsm._results[hyfsm._fsm_state][idx]
                    anno_text.append('{!r}'.format(fsm_state))
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


class ConstantPointer:
    def __init__(self, value=None):
        self._value = value

    @property
    def value(self):
        return self._value


class FuncPointer:
    def __init__(self, func):
        self.func = func

    @property
    def value(self):
        return self.func()


class StatePointer:
    def __init__(self, fsm, state_name):
        self.fsm = fsm
        self.state_name = state_name

    @property
    def value(self):
        return self.fsm.states[self.state_name]


class ResultPointer:
    def __init__(self, value=None):
        self._results = [value]

    @property
    def value(self):
        return self._results[-1]

    @value.setter
    def value(self, value):
        self._results.append(value)


class Param:
    def __init__(self, name, default=None, parent=None):
        self.name = name
        self.default = default
        self._pointer = None
        self._parent = parent

    @property
    def value(self):
        try:
            return self._pointer.value
        except AttributeError:
            return self.default

    @value.setter
    def value(self, value):
        if hasattr(value, 'value'):
            self._pointer = value
        elif hasattr(value, '__call__'):
            self._pointer = FuncPointer(value)
        else:
            self._pointer = ConstantPointer(value)

    def plot(self, row=0, col=0, xlabel=None, ylabel=None, use_cursor=True,
             **kwargs):
        kwargs['__obj__'] = self
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['use_cursor'] = use_cursor
        self._parent._plot_config.append(kwargs)

class Parameters:
    # read-only attributes except for underscore
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError("can't set attribute {}".format(repr(name)))
        object.__setattr__(self, name, value)

    def __init__(self, parent=None):
        self._params = OrderedDict()
        self._parent = parent

    def _append(self, name, default=None):
        if name in self._params:
            self._params.pop(name)
        arg = Param(name, default, parent=self._parent)
        object.__setattr__(self, name, arg)
        self._params[name] = arg
        return arg

    def __iter__(self):
        for name, param in self._params.items():
            yield name, param

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


class HyFSM(FSM):
    def __init__(self):
        self._states = {}
        super().__init__()
        self._childs = {}
        self._results = {}

        # extract inputs
        self.inputs = Parameters(self)
        try:
            names = self.__class__.inputs.split(',')
        except AttributeError:
            names = ''
        for name in names:
            self.inputs._append(name.strip())

        # extract outputs
        self.outputs = Parameters(self)
        try:
            names = self.__class__.outputs.split(',')
        except AttributeError:
            names = ''
        for name in names:
            _name = name.strip()
            if hasattr(self, _name):
                func = getattr(self, _name)
                param = self.outputs._append(_name)
                param.value = func
            else:
                msg = 'WARNING: {!r} has no (output) method {!r}'
                msg = msg.format(self.__class__.__name__, _name)
                print(msg)

        self._am = None
        self._pm = None
        self._plot_config = []
        self.is_configured = False
        #~ self._config = []
        #~ self._configure_idx = []

        self._fsm_state = Param('fsm')
        self._fsm_state.value = StatePointer(self, 'fsm')

    @property
    def _fsms(self):
        fsms = [self] if self.graph else []
        fsms.extend(self._childs.values())
        return fsms

    def plot_func(self, func_name, row=0, col=0, xlabel=None, ylabel=None,
                  use_cursor=True, **kwargs):
        if ylabel is None:
            ylabel = '{}.\n{}'
            ylabel = ylabel.format(self.__class__.__name__, func_name)
        param = Param(func_name, parent=self)
        param.value = getattr(self, func_name)
        kwargs['__obj__'] = param
        kwargs['__func_name__'] = 'add_sweep_plot'
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['use_cursor'] = use_cursor
        self._plot_config.append(kwargs)
        self._results[self._fsm_state] = []

    def plot_state(self, state_name, row=0, col=0,
                   xlabel=None, ylabel=None, use_cursor=True, **kwargs):
        param = Param(state_name, parent=self)
        param.value = StatePointer(self, state_name)
        if ylabel is None:
            ylabel = '{}.\nstates.\n{}'
            ylabel = ylabel.format(self.__class__.__name__, state_name)
        kwargs['__obj__'] = param
        kwargs['__func_name__'] = 'add_sweep_plot'
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['use_cursor'] = use_cursor
        self._plot_config.append(kwargs)
        self._results[self._fsm_state] = []

    def plot_fsm(self, row=0, col=0,
                 xlabel=None, ylabel=None, use_cursor=True, **kwargs):
        param = self._fsm_state
        if ylabel is None:
            ylabel = '{}.{}'
            ylabel = ylabel.format(self.__class__.__name__, param.name)
        kwargs['__obj__'] = param
        kwargs['__func_name__'] = 'add_state_plot'
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['use_cursor'] = use_cursor
        self._plot_config.append(kwargs)
        self._results[self._fsm_state] = []

    def plot_input(self, name, row=0, col=0,
                   xlabel=None, ylabel=None, use_cursor=True, **kwargs):
        if ylabel is None:
            ylabel = '{}.\ninputs.\n{}'
            ylabel = ylabel.format(self.__class__.__name__, name)
        param = self.inputs._params[name]
        kwargs['__obj__'] = param
        kwargs['__func_name__'] = 'add_sweep_plot'
        kwargs['row'] = row
        kwargs['col'] = col
        kwargs['xlabel'] = xlabel
        kwargs['ylabel'] = ylabel
        kwargs['use_cursor'] = use_cursor
        self._plot_config.append(kwargs)
        self._results[self._fsm_state] = []

    def configure_plots(self):
        self._am = AxesManager()
        self._pm = PlotManager(hyfsm=self)
        self._pm.am = self._am
        for fsm in self._fsms:
            for kwargs in fsm._plot_config:
                kwargs = kwargs.copy()
                obj = kwargs.pop('__obj__')
                func = getattr(self._pm, kwargs.pop('__func_name__'))
                func(obj, **kwargs)
                row = kwargs['row']
                col = kwargs['col']
                self._am.append_loc(row, col)

    def configure_main(self):
        for task in self.tasks.items:
            for func, obj, args, kwargs in task._config:
                if func.__name__ in ['plot']:
                    sig = inspect.getargspec(func)
                    func_args = dict(zip_longest(sig.args[::-1],
                                     sig.defaults[::-1]))
                    for name, val in zip(sig.args, args):
                        func_args[name] = val
                    func_args.update(kwargs)
                    row = func_args['row']
                    col = func_args['col']
                    self._am.append_loc(row, col)
                    fsm._pm.am = self._am

    def configure(self):
        for idx in self._configure_idx:
            func, obj, args, kwargs = self._config[idx]
            func(obj, *args, **kwargs)
        if self.params._pm.am is not None:
            self.params._pm.configure()

    def add_child_fsm(self, fsm, name=''):
        self._childs[name] = fsm
        if not name:
            name = fsm.__class__.__name__.lower()
        setattr(self, name, fsm)

    @property
    def state(self):
        return self._states['fsm']

    @state.setter
    def state(self, value):
        self._states['fsm'] = value

    def add_state(self, name, init=np.nan):
        self._states[name] = init

    @property
    def states(self):
        return dict(self._states, fsm=self.state)

    def next_state(self, **kwargs):
        for name, value in kwargs.items():
            self.inputs._params[name].value = value
        fsm_cache = {}
        if self.graph:
            event = self.get_active_event()
            if event:
                # make state transition and action
                newstates = self.states
                actions, newstates['fsm'] = self.graph[newstates['fsm']][event]
                for func in actions:
                    newstates.update(func())
                fsm_cache[self] = newstates
        for fsm in self._childs.values():
            child_caches = fsm.next_state()
            fsm_cache.update(child_caches)
        return fsm_cache

    def run(self, **kwargs):
        self.configure_plots()
        for name, param in self.outputs:
            self._results[param] = []
        datas = []
        names = []
        for name, values in kwargs.items():
            names.append(name)
            datas.append(values)
        if self._am is not None:
            self._pm.configure()
        for values in zip(*datas):
            dct = {name: value for name, value in zip(names, values)}
            caches = self.next_state(**dct)
            for fsm, newstates in caches.items():
                fsm._states = newstates
            for fsm in self._fsms:
                for param, datas in fsm._results.items():
                    datas.append(param.value)
            if self._am is not None:
                self._pm.plot()
        if self._am is not None:
            if self._am.fig is not None:
                self._am.fig.show()
                self._am.update()
        results = {}
        for name, param in self.outputs:
            results[name] = self._results[param]
        return results

    def out(self):
        values = {}
        for name, param in self.outputs:
            func = getattr(self, name)
            value = func()
            param._pointer.value = value
            values[name] = value
        return values
