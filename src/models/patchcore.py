"""
PatchCore: Base Expert

Captures 'Cold' anomalies (structural deviations from the memory bank).
For the Concat stream, a wider backbone (e.g., WideResNet-101) is recommended
due to increased feature dimensions.
"""

class PatchCore:
    def __init__(self, backbone_name="resnet18"):
        self.backbone_name = backbone_name
        # TODO: Initialize memory bank and feature extractor here
        pass

    def fit(self, dataloader):
        # TODO: Extract features and build the coreset memory bank
        pass

    def predict(self, images):
        # TODO: Return anomaly maps/scores and identify residuals
        pass
