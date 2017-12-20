import collections
from collections import OrderedDict
import numpy as np


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


class Argument:
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


class Inputs:
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
        arg = Argument(name, default, parent=self._parent)
        object.__setattr__(self, name, arg)
        self._params[name] = arg

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
        self.inputs = Inputs(self)
        try:
            names = self.__class__.inputs.split(',')
        except AttributeError:
            names = ''
        for name in names:
            self.inputs._append(name.strip())

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
        fsms = [self] if self.graph else []
        fsms.extend(self._childs.values())
        for fsm in fsms:
            event = fsm.get_active_event()
            if event:
                # make state transition and action
                newstates = fsm.states
                actions, newstates['fsm'] = fsm.graph[newstates['fsm']][event]
                for func in actions:
                    newstates.update(func())
                fsm_cache[fsm] = newstates
        for fsm, newstates in fsm_cache.items():
            fsm._states = newstates


if __name__ == '__main__':
    class Counter(HyFSM):
        inputs = 'start, limit'

        def __init__(self):
            super().__init__()

            self.add_state('counter', 0)
            self.add_transition('IDLE',  self.ev_start,    'COUNT')
            self.add_transition('COUNT', self.ev_running,  'COUNT', self.count)
            self.add_transition('COUNT', self.ev_finished, 'DONE')
            self.add_transition('DONE',  self.ev_stop,     'IDLE', self.reset)

        def ev_start(self):
            return self.inputs.start.value

        def ev_stop(self):
            return not self.ev_start()

        def ev_running(self):
            return self.states['counter'] < self.inputs.limit.value

        def ev_finished(self):
            return not self.ev_running()

        def count(self):
            return {'counter': self.states['counter'] + 1}

        def reset(self):
            return {'counter': 0}


    class Controler(HyFSM):
        inputs = 'limit'

        def __init__(self):
            super().__init__()

            self.add_transition('IDLE', 'COUNT')
            self.add_transition('COUNT', self.ev_counter_done, 'DONE')
            self.add_transition('DONE', 'IDLE')

            self.add_child_fsm(Counter(), name='counter')
            self.counter.inputs.start.value = self.count
            self.counter.inputs.limit.value = self.inputs.limit

        def ev_counter_done(self):
            return self.counter.states['fsm'] == 'DONE'

        def count(self):
            return self.states['fsm'] == 'COUNT'


    ctrl = Controler()
    print(ctrl.states, ctrl.counter.states)
    for n in range(15):
        ctrl.next_state(limit=3)
        print(ctrl.states, ctrl.counter.states)
