# -*- coding: utf-8 -*-

__author__ = 'olegtroitskiy'

import math


def get_first(iterable):
    """
    Функция помогает получить доступ к первому элементу объектов set()
    :param iterable: Любой объект, с которым работает itarable
    :return: Первый элемент
    """
    try:
        return iter(iterable).next()
    except StopIteration:
        return None

def get_by_name(iterable, name):
    """
    Функция возвращает элемент объектов set(), с заданным уникальным именем
    
    :param iterable: Любой объект, с которым работает itarable
    :param name: Глобальное имя объекта
    
    :return: Элемент
    """
    try:
        for i in iterable:
            if i.hash == name:
                return i
    except:
        return None

def get_by_pos(iterable, position):
    """
    Функция возвращает элемент объектов set(), с заданной уникальной позицией

    :param iterable: Любой объект, с которым работает itarable
    :param position: Порядеовый номер объекта (проверяется self.position)

    :return: Элемент
    """
    try:
        for i in iterable:
            if i.position == position:
                return i
    except:
        return None


def sigmoid(x):
    """
    Функция активации - тангенс
    :param x:
    :return:
    """
    return math.tanh(x)


# -------------------------------------------------------------------------
def dsigmoid(y):
    """
    Производная от функции Sigmoid
    http://www.math10.com/en/algebra/hyperbolic-functions/hyperbolic-functions.html

    :param y:
    :return:
    """
    return 1 - y ** 2
