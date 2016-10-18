# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

import math
import datetime

from Net import Net

# import pandas as pd
# import numpy as np

# import sys
# sys.path.append('../../ProgressPCT')
# from ProgressPCT import ProgressPCT

# progress_bar_length = 50

layers = 2
neurons = 7
epoch = 10000
l_rate = 0.01
m_rate = 0.005
printcount = 100

name_f = 'cos1'

# ------------------------------------------------- создаем обучающий датасет
pat = []
for i in xrange(-180, 180, 1):
    pat.append([dict(input_0=i/10.0), dict(output_0=math.cos(i/10.0))])

simple_net = Net()

simple_net.define(connectiontype='fullconnect', inputs=1, hiddenlayers=layers, neurons=neurons, outputs=1)

prog_start_time = datetime.datetime.now()

simple_net.train_pattern(pattern=pat,
                         max_iterations=epoch,
                         print_error_times=printcount,
                         n_speed=l_rate,
                         m_speed=m_rate)

print('Время выполнения: {}'.format(datetime.datetime.now() - prog_start_time))


import pickle

error_string = str(simple_net.get_pattern_error())

f = open('net_{}_state.pckl'.format(name_f), 'w')
pickle.dump(simple_net, f)
f.close()

f = open('net_{}_pattern.pckl'.format(name_f), 'w')
pickle.dump(pat, f)
f.close()

print '\n\n Ошибка на паттерне: {}'.format(error_string)