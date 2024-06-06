from app.api.v1.schemas import IPPFeatures


ipp_feature_keys = IPPFeatures.model_fields.keys()
ipp_feature_desctiprion = list(map(lambda feature: feature.description, IPPFeatures.model_fields.values()))
ipp_features = {key: description for key, description in zip(ipp_feature_keys, ipp_feature_desctiprion)}

FeatureMapper = {
    "Индекс промышленного производства": ipp_features
}
