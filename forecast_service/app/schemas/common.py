from enum import Enum


class ConfidenceIntervalEnum(str, Enum):
    """Доверительный интервал"""

    low = '90%'
    medium = '95%'
    hight = '99%'


class ReadyOnModels(str, Enum):
    """Индексы для которых готовы модели машинного обучения"""

    ipp = 'ipp'
    ipc = 'ipc'
    ort = 'ort'
