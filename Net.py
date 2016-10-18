# -*- coding: utf-8 -*-

__author__ = 'Олег Троицкий'

from Layer import Layer
from Neuron import Neuron
import Utils


class Net:
    """ Нейронная сеть """

    # -----------------------------------------------------------------------------
    def __init__(self):

        # ------------------------------------------ Создаем контейнеры для входных, промежуточных и выходных слоев
        self.inp_layers = set()                     # "контейнер", содержащий не повторяющиеся элементы в случ. порядке
        self.layers = set()
        self.out_layers = set()

        self.hidden_layers_total = 0

        self.layers_list = []

        self.epoch = 0

        self.last_sample_error = 0.0
        self.pattern_error = 0.0

    # -----------------------------------------------------------------------------
    def add_inp_layer(self, layer):
        """ Добавляем в сеть входной слой """

        # ----------------------------- Проверяем, что добавляем слой и что в сети уже нет слоя с таким названиеми
        if isinstance(layer, Layer) and layer.name not in self.layers_list:

            if layer not in self.inp_layers:
                self.inp_layers.add(layer)

                layer.owner = self

    # -----------------------------------------------------------------------------
    def add_layer(self, layer, layer_position):
        """ Добавляем в сеть Hidden layer """

        # ----------------------------- Проверяем, что добавляем слой и что в сети уже нет слоя с таким названиеми
        if isinstance(layer, Layer):

            if layer.name not in self.layers_list:

                # TODO добавить проверку – в сети уже может быть слой с таким-же порядковым номером
                if layer.position == None:

                    if layer not in self.layers:
                        self.layers.add(layer)

                        self.layers_list.append(layer.name)

                        self.hidden_layers_total += 1

                        layer.owner = self
                        layer.position = layer_position

                else:
                    print 'Ошибка. Слой уже имеет порядковый номер'

            else:
                print 'Ошибка. В сети уже есть слой с именем:', layer.name

    # -----------------------------------------------------------------------------
    def add_out_layer(self, layer):
        """ Добавляем в сеть Input layer """

        # ----------------------------- Проверяем, что добавляем слой и что в сети уже нет слоя с таким названиеми
        if isinstance(layer, Layer) and layer.name not in self.layers_list:

            if layer not in self.out_layers:
                self.out_layers.add(layer)

                layer.owner = self

    def get_epoch(self):
        return self.epoch

    def get_hidden_layers(self):
        return len(self.layers)

    def get_hidden_neurons(self):
        return len(Utils.get_first(self.layers).neurons)

    def get_input_neurons(self):
        return len(Utils.get_first(self.inp_layers).neurons)

    def get_output_neurons(self):
        return len(Utils.get_first(self.out_layers).neurons)

    def get_last_sample_error(self):
        return self.last_sample_error

    def get_pattern_error(self):
        return self.pattern_error

    def forward_pass(self, input_signal):
        """
        Прямой проход

        :param input_signal: Сигнал на входе сети
        :return: Возвращаем сигналы на выходе сети
        """

        out_signals = {}

        # Прописываем входные сигналы во входной слой

        if len(self.inp_layers) > 0:

            l = Utils.get_first(self.inp_layers)                    # Берем входной слой

            if len(input_signal) == len(l.neurons)-1:

                for n in l.neurons:

                    if not n.bias_flag:
                        n.signal = input_signal[n.hash]

            else:
                print 'Ошибка. Указано неверное количество нейронов на входе сети'

            # ------------------------------------------------------------ Промежуточные слои
            for i in xrange(self.hidden_layers_total):

                l = Utils.get_by_pos(self.layers, position=i)

                # print '\n\n', 'Слой:', l.name

                for n in l.neurons:

                    tot_sum = 0.0

                    for inp_neuron in n.inputs:

                        tot_sum += n.weights[inp_neuron.hash] * inp_neuron.signal

                    n.signal = Utils.sigmoid(tot_sum)

                    # print 'Нейрон:', n.name, '    Сигнал на выходе:', tot_sum, '    Функция активации:', n.signal

            # ------------------------------------------------------------ Input layer
            if len(self.out_layers) > 0:

                l = Utils.get_first(self.out_layers)

                # print '\n\n', 'Слой:', l.name

                for n in l.neurons:

                    tot_sum = 0.0

                    for inp_neuron in n.inputs:

                        tot_sum += n.weights[inp_neuron.hash] * inp_neuron.signal

                    n.signal = Utils.sigmoid(tot_sum)

                    # print 'Нейрон:', n.name, '    Сигнал на выходе:', tot_sum, '    Функция активации:', n.signal

                    out_signals[n.hash] = n.signal

            else:
                print 'Ошибка. В сети нет выходного слоя'

        else:
            print 'Ошибка. В сети нет входного слоя'

        return out_signals

    def backward_pass(self, targets, n_speed, m_speed):
        """
        Обратный проход.
        Считаем ошибки на каждом нейроне и изменения весов связей

        :param targets: целевые значения на выходах сети (словарик - для каждого имени нейрона указан его таргет)
        :param n_speed: скорость обучения
        :param m_speed: сила момента (продолжение сдвига веса, рассчитанного на пред. шаге)
        :return: считает общую ошибку (качество обучения сети)
        """

        # ------------------------------------ Проходим по нейронам выходного слоя

        net_error = 0.0                                     # Ошибка на выходе сети

        l = Utils.get_first(self.out_layers)                # Обращаемся к первому из выходных слоев

        for n in l.neurons:

            # -------------------------------- Считаем ошибку: разница между целевым значением на выходе и рассчитанным
            n.error = targets[n.hash] - n.signal

            # -------------------------------- Получаем необходимый сдвиг на n-ом нейроне выходного слоя путем умножения
            # -------------------------------- ошибки этого нейрона на производную от ф-ии активации по рассчит. знач.
            n.output_delta = n.error * Utils.dsigmoid(n.signal)

            for inp_neuron in n.inputs:
                change = n.output_delta * inp_neuron.signal
                n.weights[inp_neuron.hash] += n_speed * change + m_speed * n.prev_change[inp_neuron.hash]

                n.prev_change[inp_neuron.hash] = change

            # Добавляем к общей ошибке сети ошибку на данном нейроне.
            # Вычисляем общую ошибку сети через евклидово расстояние.
            net_error += n.error ** 2

        # ------------------------------------ Проходим по скрытым слоям в обратном порядке

        for i in reversed(xrange(self.hidden_layers_total)):

            l = Utils.get_by_pos(self.layers, i)                # Обращаемся к первому из выходных слоев

            for n in l.neurons:

                n.error = 0.0

                # ---------------------------- Для каждого нейрона промеж. слоя считаем ошибку. Она зависит от кол-ва
                # ---------------------------- вых. нейронов n_out, с которыми он связан. Чем больше связ. вых. нейронов
                # ---------------------------- с сильно неправильными значениями, тем сильнее в итоге изменется
                # ---------------------------- вес n-го нейрона

                for n_out in n.outputs:

                    # ------------------------ Накапливаем ошибку: умножаем вес n-го нейрона промежуточного слоя
                    # ------------------------ на нужный нам сдвиг на связанном n_out нейроне
                    n.error += n_out.output_delta * n_out.weights[n.hash]

                n.output_delta = n.error * Utils.dsigmoid(n.signal)

                # # -------------------------------- Запоминаем ошибку: разница между целевым значением на выходе
                # # -------------------------------- и рассчитанным
                # n.error = targets[n.hash] - n.signal
                #
                # # -------------------------------- Получаем необходимый сдвиг на k-ом нейроне выходного слоя путем
                # # -------------------------------- умножения ошибки этого нейрона на производную от ф-ии активации
                # # -------------------------------- по рассчит. знач.
                # n.output_delta = n.error * Utils.dsigmoid(n.signal)

                for inp_neuron in n.inputs:
                    change = n.output_delta * inp_neuron.signal
                    n.weights[inp_neuron.hash] += n_speed * change + m_speed * n.prev_change[inp_neuron.hash]

                    n.prev_change[inp_neuron.hash] = change

        self.epoch += 1

        #TODO Проверить, здесь половина или корень?
        self.last_sample_error = net_error ** 0.5

        return self.last_sample_error

    # -----------------------------------------------------------------------------
    def test(self, input_signal, targets):
        """
        Одиночный вариант

        Выводит на экран ответ сети на входные значения, плюс те значения, которые на самом деле должны быть
        :param input_signal: Список значений на входе нейронной сети
        :param targets: целевые значения на выходах сети (словарик - для каждого имени нейрона указан его таргет)
        :return:
        """

        print 'На входе сети:', input_signal, '-->', self.forward_pass(input_signal=input_signal), '\tЦель', targets

    # -----------------------------------------------------------------------------
    def test_pattern(self, pattern):
        """
        Мульти вариант

        Выводит на экран ответ сети на входные значения, плюс те значения, которые на самом деле должны быть
        :param pattern: Паттерн из списков значений на входе и ответов на выходе
        :return:
        """

        for p in pattern:

            input_signal = p[0]
            targets = p[1]

            out_signal = self.forward_pass(input_signal=input_signal)

            print 'На входе сети:', input_signal, '-->', out_signal, '\tЦель', targets

        return out_signal

    # -----------------------------------------------------------------------------
    def train(self, input_signal, targets, max_iterations=1000, print_error_times=10, n_speed=0.1, m_speed=0.1):
        """

        Тренируем сеть на 1 примере (сигнал на входе дает сигнал на выходе

        :param input_signal: Словарик значений на входе нейронной сети
        :param targets: Словарик значений на выходе нейронной сети
        :param max_iterations: Количество итераций обучения
        :param print_error_times: Сколько раз выводить ошибку при вычислениях
        :param n_speed: Сила обучения
        :param m_speed: Сила момента (сдвиг весов в направлении предыдущего шага)
        :return:
        """

        separate = round(max_iterations / print_error_times + 0.5)

        for i in xrange(max_iterations):

            self.forward_pass(input_signal=input_signal)              # Считаем сигнал на выходе сети (прямой проход)
            error = self.backward_pass(targets=targets, n_speed=n_speed, m_speed=m_speed)

            # -------------------------------- Выводим на экран значение итоговой ошибки сети
            if i % separate == 0:
                print 'Общая ошибка: ', error

        self.epoch += 1

        # ------------------------------------ В конце выводим ответ сети на входные сигналы
        self.test(input_signal=input_signal, targets=targets)

    # -----------------------------------------------------------------------------
    def train_pattern(self, pattern, max_iterations=1000, print_error_times=10, n_speed=0.1, m_speed=0.02):
        """

        Тренируем сеть на паттерне примеров (различные варианты сигналов на входе должны давать различные
        варианты сигналов на выходе

        :param pattern: Список значений на входах и выходах нейронной сети для обучения
        :param max_iterations: Количество итераций обучения
        :param print_error_times: Сколько раз выводить ошибку привычислениях
        :param n_speed: Сила обучения
        :param m_speed: Сила момента (сдвиг весов в направлении предыдущего шага)
        :return:
        """

        separate = round(max_iterations / print_error_times)

        pattern_dimention = len(pattern)

        for i in xrange(max_iterations):

            pattern_error = 0.0

            for p in pattern:
                input_signal = p[0]
                targets = p[1]

                self.forward_pass(input_signal=input_signal)          # Считаем сигнал на выходе сети (прямой проход)
                error = self.backward_pass(targets=targets, n_speed=n_speed, m_speed=m_speed)

                pattern_error += error

            self.pattern_error = pattern_error / pattern_dimention

            # -------------------------------- Выводим на экран значение итоговой ошибки сети
            if i % separate == 0:
                # self.last_sample_error()
                # print 'Ошибка сети на последнем сэмпле: {:12.8} средняя ошибка на паттерне: {:12.8}'.format(
                #     error,
                #     pattern_error / pattern_dimention)

                print 'Средняя ошибка на паттерне: {:12.8f}'.format(self.pattern_error)

        # ------------------------------------ В конце выводим ответ сети на входные сигналы
        self.test_pattern(pattern=pattern)

    # -----------------------------------------------------------------------------
    def define(self, connectiontype, inputs, hiddenlayers, neurons, outputs):
        """
        Определяем нейронную сеть в автоматическом режиме.

        :param connectiontype:  вариант соединения нейронов между слоями
                fullconnect  -   между каждыми соседними слоями все нейроны соеденены каждый с каждым

        :param inputs:          кол-во нейронов во входном слое
        :param hiddenlayers:    кол-во промежуточных слоев
        :param neurons:         кол-во нейронов в каждом из промежуточных слоев
        :param outputs:         кол-во нейронов в выходном слое

        :return:
        """

        if connectiontype == 'fullconnect':

            # ------------------------------------------ Создаем входной слой
            input_layer = Layer(name='Input layer')

            self.add_inp_layer(layer=input_layer)

            bias = Neuron(name='Bias', bias_flag=True, globalname='INB')
            input_layer.add_neuron(bias)

            input_neuron = [None] * inputs

            for n in xrange(inputs):
                input_neuron[n] = Neuron(name=str(n), globalname='input_' + str(n))
                input_layer.add_neuron(input_neuron[n])

            # ------------------------------------------ Промежуточные слои
            hidden_layer = [None] * hiddenlayers

            for i in xrange(hiddenlayers):
                hidden_layer[i] = Layer(name='Hidden layer ' + str(i))

                self.add_layer(layer=hidden_layer[i], layer_position=i)

                for n in xrange(neurons):

                    hidden_neuron = [None] * neurons

                    hidden_neuron[n] = Neuron(name=str(n), globalname='hidden_' + str(i) + str(n))
                    hidden_layer[i].add_neuron(hidden_neuron[n])

            # ------------------------------------------ Создаем выходной слой
            output_layer = Layer(name='Output layer')

            # ------------------------------------------ Добавляем нейроны в выходной слой
            self.add_out_layer(layer=output_layer)

            output_neuron = [None] * outputs

            for n in xrange(outputs):
                output_neuron[n] = Neuron(name=str(n), globalname='output_' + str(n))
                output_layer.add_neuron(output_neuron[n])

            # ------------------------------------------- Соединяем ранее созданные нейроны между собой

            # Каждый нейрон первого скрытого слоя соединяем с нейронами входного слоя
            i = Utils.get_first(self.inp_layers)    # Обращаемся к первому из входных слоев
            h = Utils.get_by_pos(self.layers, 0)    # Обращаемся к первому из скрытых слоев

            for hid_n in h.neurons:                             # Идем по каждому нейрону первого промежуточного слоя

                for inp_n in i.neurons:                         # Идем по каждому нейрону входного слоя
                    hid_n.set_inp_link(inp_n)                   # Соединяем вход нейрона из скрытого слоя с нейроном из
                                                                # входного слоя

            # Проходим по остальным скрытым слоям
            if hiddenlayers > 1:

                prev_layer = h                                  # Запонминаем первый промежуточный слой как предыдущий

                for i in xrange(1, hiddenlayers):               # Перемещаемся по каждому следующему промежуточному слою

                    cur_layer = Utils.get_by_pos(self.layers, i)        # Запоминаем слой

                    for n in cur_layer.neurons:

                        for p in prev_layer.neurons:

                            n.set_inp_link(p)

                    prev_layer = cur_layer

                cur_layer = Utils.get_first(self.out_layers)    # Берем выходной слой

                for n in cur_layer.neurons:

                    for p in prev_layer.neurons:
                        n.set_inp_link(p)


                print 'Нейроны соеденены'

