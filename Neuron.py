# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

import random

class Neuron:
    """ Нейрон """

    # -----------------------------------------------------------------------------
    def __init__(self, globalname, name=None, bias_flag=False, ):

        self.name = name
        self.hash = globalname              # Уникальное название. Используется для словарика с весами

        self.inputs = set()
        self.outputs = set()

        self.weights = {}                   # Храним веса по каждой связи (словарик по имени нейрона)
        self.prev_change = {}               # Храним предыдущее изменение веса по каждой связи (нужно для для момента)

        self.owner = None

        # --------------------------------- Нейрон BIAS должен всегда давать единицу
        if bias_flag:
            self.signal = 1.0
        else:
            self.signal = 0.0

        self.error = 0.0                    # Величина ошибки на выходе нейрона

        self.output_delta = 0.0             # Необходимый сдвиг на выходе нейрона

        self.bias_flag = bias_flag

    # -----------------------------------------------------------------------------
    def set_inp_link(self, neuron):
        """ Связываем вход текущего нейрона с выходом нейрона из предыдущего слоя """

        if isinstance(neuron, Neuron):

            if neuron not in self.inputs:
                self.inputs.add(neuron)
                self.weights[neuron.hash] = random.uniform(-1, 1)
                self.prev_change[neuron.hash] = 0.0

                # ----------------------------------- Прописываем обратную связь
                neuron.set_out_link(self)

            else:
                print 'Отказ. Попытка повторно прописать связь нейрона: ', self.name, 'с нейроном:', neuron.name

    # -----------------------------------------------------------------------------
    def set_out_link(self, neuron):
        """ Связываем выход текущего нейрона со входом нейрона из следующего слоя """

        if isinstance(neuron, Neuron):

            if neuron not in self.outputs:
                self.outputs.add(neuron)

            else:
                print 'Отказ. Попытка повторно прописать связь нейрона:', self.name, 'с нейроном:', neuron.name

    def __str__(self):
        """ Выводим информацию о нейроне """

        str_align = '        '

        print '\n', str_align, 'Нейрон:', self.name
        print str_align, 'Глобальное имя:', self.hash
        print str_align, 'Слой:', self.owner.name
        print str_align, 'Сигнал:', self.signal
        print str_align, 'Ошибка:', self.error
        print str_align, '-------------------------------------------\n'

        print str_align, 'Входы'

        for inp_neuron in self.inputs:
            print str_align, '    ', inp_neuron.name, '    Вес: ', self.weights[inp_neuron.hash]

        print str_align, '-------------------------------------------'
        print str_align, 'Выходы\n'

        for i in self.outputs:
            print str_align, '    ', inp_neuron.name

        print str_align, '-------------------------------------------\n'

        return ''
