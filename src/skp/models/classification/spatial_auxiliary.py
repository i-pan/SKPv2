"""Config-wired spatial classification with an optional shared-encoder decoder."""

from timm import create_model
from torch import nn

from skp.models.normalization import normalize_input
from skp.models.spatial import AuxiliaryDecoder, SpatialLogitHead


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Architecture-specific stride changes belong in the downstream encoder factory.
        kwargs = dict(cfg.get("encoder_kwargs", {}) or {})
        self.encoder = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            in_chans=cfg.num_input_channels,
            **kwargs,
        )
        channels = self.encoder.feature_info.channels()
        self.classifier = SpatialLogitHead(
            channels[-1],
            cfg.num_classes,
            pooling=cfg.get("spatial_pooling", "max"),
            temperature=cfg.get("spatial_temperature", 1.0),
        )
        output_channels = cfg.get("auxiliary_output_channels", 0)
        self.decoder = None
        if output_channels:
            self.decoder = AuxiliaryDecoder(
                channels, cfg.require("auxiliary_decoder_channels"), output_channels
            )
        self.criterion = None

    def forward(
        self, batch, return_loss=False, return_features=False, return_segmentation=False
    ):
        features = self.encoder(normalize_input(batch["x"], self.cfg))
        out = self.classifier(features[-1])
        if return_features:
            out["features"] = features
        if return_segmentation or (return_loss and self.decoder is not None):
            if self.decoder is None:
                raise ValueError("Segmentation requested without an auxiliary decoder")
            out["segmentation_logits"] = self.decoder(features, batch["x"].shape[-2:])
        if return_loss:
            if self.criterion is None:
                raise ValueError("Set a criterion before requesting loss")
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss):
        self.criterion = loss
