from enum import Enum


class ConfidenceIntervalEnum(str, Enum):
    """Доверительный интервал"""

    low = '90%'
    medium = '95%'
    hight = '99%'

