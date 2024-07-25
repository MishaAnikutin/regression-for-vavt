from .io.response import ForecastResponse, FeatureResponse, IPPFeaturesResponse, IPCFeaturesResponse, FeaturesResponse
from .io.request import FeatureRequest, IPPRequestCB, BaseRequest

from .ml.features import Feature, IPPFeatures
from .ml.params import BaseHyperparameters, RNNHyperparameters, CatBoostHyperparameters
from .ml.scores import ModelScore

from .common import ConfidenceIntervalEnum, ReadyOnModels


# Соответствие между готовыми моделями и их признаками
IndexFeaturesMapper = {
    ReadyOnModels.ipp: IPPFeaturesResponse,
    ReadyOnModels.ipc: IPCFeaturesResponse
}


__all__ = [
    'IPCFeaturesResponse',
    'IPPFeaturesResponse',

    'BaseRequest',
    'IPPRequestCB',
    'ForecastResponse',

    'FeatureRequest',
    'FeatureResponse',
    'FeaturesResponse',

    'Feature',
    'ModelScore',
    'IPPFeatures',
    'RNNHyperparameters',
    'CatBoostHyperparameters',
    'BaseHyperparameters',

    'ConfidenceIntervalEnum',
    'ReadyOnModels',
    'IndexFeaturesMapper'
]
