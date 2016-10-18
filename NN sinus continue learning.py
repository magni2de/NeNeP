# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

import math
import datetime

from Net import Net
import pickle

l_rate = 0.01
m_rate = 0.005
epoch = 30000
printcount = 100

name = 'f(x) = tan(x)'
name_f = 'cos1'

f = open('net_{}_state.pckl'.format(name_f), 'r')
simple_net = pickle.load(f)
f.close()

layers = simple_net.get_hidden_layers()
neurons = simple_net.get_hidden_neurons()

f = open('net_{}_pattern.pckl'.format(name_f), 'r')
pat = pickle.load(f)
f.close()

error_string = str(simple_net.get_pattern_error())
print '\n\n Ошибка на паттерне в начале обучения: {}\n'.format(error_string)

prog_start_time = datetime.datetime.now()

simple_net.train_pattern(pattern=pat,
                         max_iterations=epoch,
                         print_error_times=printcount,
                         n_speed=l_rate,
                         m_speed=m_rate)

print('Время выполнения: {}'.format(datetime.datetime.now() - prog_start_time))

error_string = str(simple_net.get_pattern_error())

f = open('net_{}_state.pckl'.format(name_f), 'w')
pickle.dump(simple_net, f)
f.close()

print '\n\n Ошибка на паттерне в конце обучения: {}'.format(error_string)