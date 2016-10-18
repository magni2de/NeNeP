# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

from Neuron import Neuron


class Layer:
    """ Слой """

    def __init__(self, name=None):

        self.position = None
        self.name = name

        self.neurons_total = 0

        self.owner = None

        self.neurons = set()

    # -----------------------------------------------------------------------------
    def add_neuron(self, neuron):
        """ Добавляем в слой нейрон """

        if isinstance(neuron, Neuron):

            if neuron not in self.neurons:
                self.neurons.add(neuron)

                neuron.owner = self

                self.neurons_total += 1


# class inpLayer(Layer):
#     """ Слой """
#
#     # def __init__(self, name=None):
#     #
#     #     Layer.__init__(self, name=name)
#
#     # -----------------------------------------------------------------------------
#     def add_neuron(self, neuron):
#         """ Добавляем в сеть промежуточный слой """
#
#         if isinstance(neuron, Neuron):
#
#             if neuron not in self.neurons:
#                 self.neurons.add(neuron)
#
#                 neuron.owner = self
