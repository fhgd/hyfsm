from hyfsm import HyFSM
import matplotlib.pylab as plt


class Counter(HyFSM):
    """A simple counter FSM with two inputs:

        cmd_start:    Command for starting the counter
                      (True/False)

        limit:        maximum counting limit at which the counter stops
                      (integer)

    and one output:

        counter:      the current counter-value
    """
    # list all input and output parameters as a comma separated string
    inputs = 'cmd_start, limit'
    outputs = 'counter'

    def __init__(self):
        super().__init__()

        # add new discrete state variable and set initial value
        self.add_state('counter', 0)

        # define FSM graph
        self.add_transition('IDLE',     self.ev_start,    'COUNTING', self.count)
        self.add_transition('COUNTING', self.ev_running,  'COUNTING', self.count)
        self.add_transition('COUNTING', self.ev_finished, 'DONE')
        self.add_transition('DONE',     self.ev_stop,     'IDLE', self.reset)

    def ev_start(self):
        return self.inputs.cmd_start.value

    def ev_stop(self):
        return not self.ev_start()

    def ev_running(self):
        return self.states['counter'] < self.inputs.limit.value

    def ev_finished(self):
        return not self.ev_running()

    # action
    def count(self):
        return {'counter': self.states['counter'] + 1}

    # action
    def reset(self):
        return {'counter': 0}

    # output
    def counter(self):
        return self.states['counter']


class Controler(HyFSM):
    """A simple controler FSM which controls a counter as a child-FSM.

    Input:
        limit:          maximum limit value for the counter

    Outputs:
        cmd_count:      command for starting the child-counter-FSM
        child_counter:  counter value of child-FSM
    """
    # list all input and output parameters as a comma separated string
    inputs = 'limit'
    outputs = 'cmd_count, child_counter'

    def __init__(self):
        super().__init__()

        # define graph of parent-FSM
        self.add_transition('IDLE', 'COUNT')
        self.add_transition('COUNT', self.ev_counter_done, 'DONE')
        self.add_transition('DONE', 'IDLE')

        # create child-FSM
        self.add_child_fsm(Counter(), name='counter')

        # connect input parameters of child-FSM
        self.counter.inputs.cmd_start.value = self.cmd_count
        self.counter.inputs.limit.value = self.inputs.limit

    def ev_counter_done(self):
        # observe states of child-FSM
        return self.counter.states['fsm'] == 'DONE'

    # output
    def cmd_count(self):
        return self.states['fsm'] == 'COUNT'

    # output
    def child_counter(self):
        return self.counter.states['counter']


# create Controler-FSM
ctrl = Controler()

# configure plots
ctrl.plot_func('cmd_count', row=0, xlabel='')
ctrl.plot_func('ev_counter_done', row=1, xlabel='')
ctrl.counter.plot_state('counter', row=2, xlabel='')
ctrl.counter.plot_input('cmd_start', row=3)
ctrl.configure_plots()

# process an input stream with the hierachical stacked
# controler-counter-FSM
values = [3] * 20
results = ctrl.run(limit=values)

# inspect the results
print('\n** Click on the data points seeing their values and fsm-states **\n')
print('output values:')
for name, values in results.items():
    print('  {:>13}:  {}'.format(name, values))
