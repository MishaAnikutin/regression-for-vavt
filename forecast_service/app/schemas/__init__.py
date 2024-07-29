from .io.response import ForecastResponse, FeatureResponse, IPPFeaturesResponse, IPCFeaturesResponse, FeaturesResponse
from .io.request import FeatureRequest, IPPRequestCB, BaseRequest, IPCRequestCB

from .ml.features import Feature, IPPFeatures
from .ml.params import BaseHyperparameters, RNNHyperparameters, CatBoostHyperparameters
from .ml.scores import ModelScore

from .common import ConfidenceIntervalEnum, ReadyOnModels


# Соответствие между готовыми моделями и их признаками
IndexFeaturesMapper = {
    ReadyOnModels.ipp: IPPFeaturesResponse(),
    ReadyOnModels.ipc: IPCFeaturesResponse()
}

