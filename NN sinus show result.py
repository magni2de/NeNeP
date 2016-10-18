# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

import pickle
import pylab as pl

name = 'f(x) = sin(x)'
name_f = 'sin3'

f = open('net_{}_state.pckl'.format(name_f), 'r')
simple_net = pickle.load(f)
f.close()

layers = simple_net.get_hidden_layers()
neurons = simple_net.get_hidden_neurons()
epoch = simple_net.get_epoch()

error_string = str(simple_net.get_pattern_error())

net_desc = 'Net {}: Error {}, Layers {:d}, neurons {:d}, epoch: {:d}'.format(name, error_string, layers, neurons, epoch)

f = open('net_{}_pattern.pckl'.format(name_f), 'r')
pat = pickle.load(f)
f.close()

pl.figure(figsize=(15, 3))

pl.subplot(1, 1, 1)
pl.title(net_desc)

pl.xlabel('x')
pl.ylabel('sin(x)')

pl.xlim([-18.1, 18.1])
pl.ylim([-1.5, 1.5])

x1 = []
y1 = []
out1 = []

for i in xrange(len(pat)):
    x1.append(pat[i][0].get('input_0'))
    y1.append(pat[i][1].get('output_0'))
    out1.append(simple_net.forward_pass({'input_0': x1[i]}).get('output_0'))

pl.plot(x1, y1, c='green')
pl.plot(x1, out1, c='blue')

pl.grid(True)

# -------------------------------------

pl.show()

