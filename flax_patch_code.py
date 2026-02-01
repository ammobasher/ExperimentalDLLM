  FLAX_PATCH_MAP = {
    "_Conv": {"features": int, "kernel_size": Any},
    "FlaxUpsample2D": {"in_channels": int},
    "FlaxDownsample2D": {"in_channels": int},
    "FlaxResnetBlock2D": {"in_channels": int, "out_channels": int},
    "FlaxAttentionBlock": {"channels": int},
    "FlaxDownEncoderBlock2D": {"in_channels": int, "out_channels": int},
    "FlaxUpEncoderBlock2D": {"in_channels": int, "out_channels": int},
    "FlaxUpDecoderBlock2D": {"in_channels": int, "out_channels": int},
    "FlaxUNetMidBlock2D": {"in_channels": int},
  }

  if cls.__name__ in FLAX_PATCH_MAP:
      from collections import OrderedDict
      missing_fields = FLAX_PATCH_MAP[cls.__name__]
      new_anns = OrderedDict()
      for k, v in missing_fields.items():
          new_anns[k] = v
      for k, v in list(cls_annotations.items()):
          if k not in new_anns:
              new_anns[k] = v
      cls_annotations.clear()
      cls_annotations.update(new_anns)

  if "shared_weights" in cls_annotations:
      del cls_annotations["shared_weights"]

  transformed_cls: type[M] = dataclasses.dataclass(cls, **kwargs)
