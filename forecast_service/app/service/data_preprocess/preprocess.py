from dataclasses import dataclass
from datetime import date 
import pandas as pd


@dataclass
class TimeSeries:
    """Класс с предобработкой временного ряда

    Для инициализации требует данные и их периоды времени


    """
    values: list
    dates: list[date]

    def __post_init__(self):
        if len(self.values) != len(self.dates):
            raise ValueError(f'value и dates Должны быть одной длины, но имеем: {len(self.dates) =}, {len(self.values) =}')

    def _create_df(self):
        tmp = pd.DataFrame({"date": self.dates, "value": self.values})

        tmp.date = tmp.date.apply(
            lambda x: date(
                day=1,
                month=int(x.split('.')[1]),
                year=int(x.split('.')[2])
            )
        )

        return tmp.sort_values(by='date')

    def days_to_months(self, method='mean') -> 'TimeSeries':
        """
        Ежедневные данные приводятся к ежемесячным по методу
        (среднее за месяц или другое, определенное в pandas.agg)
        """

        tmp = self._create_df()

        tmp['month-date'] = tmp.date.apply(lambda d: date(day=1, month=d.month, year=d.year))

        tmp = tmp.groupby('month-date', as_index=False).agg({'value': method})

        return TimeSeries(values=list(tmp['value']), dates=list(tmp['month-date']))

