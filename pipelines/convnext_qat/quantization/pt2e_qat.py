"""Graph-mode PT2E QAT for the backbone body with an FP32 FPN/head boundary."""

from __future__ import annotations

import copy
import io
import sysconfig
from collections import OrderedDict
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch import nn
from torch.export import Dim, export


def _normalized_version(value):
    return tuple(int(part) for part in value.split("+", 1)[0].split(".")[:3])


def _collect_pt2e_fake_quantizers(model):
    required_methods = {
        "enable_observer", "disable_observer", "enable_fake_quant", "disable_fake_quant",
    }
    return [
        module for module in model.modules()
        if required_methods.issubset(dir(module))
    ]


def validate_pt2e_schedule(total_epochs, observer_warmup_epochs, observer_freeze_epochs):
    """Validate a QAT phase schedule before expensive model construction starts."""
    values = {
        "total_epochs": int(total_epochs),
        "observer_warmup_epochs": int(observer_warmup_epochs),
        "observer_freeze_epochs": int(observer_freeze_epochs),
    }
    if values["total_epochs"] <= 0:
        raise ValueError("PT2E total_epochs must be positive")
    if values["observer_warmup_epochs"] < 0 or values["observer_freeze_epochs"] < 0:
        raise ValueError("PT2E observer warmup/freeze epochs cannot be negative")
    if values["observer_warmup_epochs"] + values["observer_freeze_epochs"] > values["total_epochs"]:
        raise ValueError("PT2E observer warmup + freeze epochs exceed total epochs")
    if values["observer_freeze_epochs"] == values["total_epochs"]:
        raise ValueError(
            "PT2E cannot freeze observers for every epoch: activation ranges would never be "
            "calibrated. Set observer_freeze_epochs=0 for a one-epoch smoke run."
        )
    return values


def pt2e_qat_phase(epoch, total_epochs, observer_warmup_epochs, observer_freeze_epochs):
    """Return the deterministic QAT phase for a zero-based epoch index."""
    values = validate_pt2e_schedule(
        total_epochs, observer_warmup_epochs, observer_freeze_epochs,
    )
    epoch = int(epoch)
    if not 0 <= epoch < values["total_epochs"]:
        raise ValueError(f"PT2E epoch index {epoch} is outside [0, {values['total_epochs']})")
    if epoch < values["observer_warmup_epochs"]:
        return "observer_warmup"
    if epoch >= values["total_epochs"] - values["observer_freeze_epochs"]:
        return "frozen"
    return "full"


def _torchao_pt2e():
    if _normalized_version(torch.__version__) < (2, 11, 0):
        raise RuntimeError(
            "PT2E QAT in this repo needs torch >= 2.11.0. "
            f"Found torch {torch.__version__}. "
            "Kaggle's default torch 2.10 image is too old for the torchao PT2E path here; "
            "upgrade torch/torchvision first, then reinstall torchao and rerun the notebook."
        )
    try:
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
        from torchao.quantization.pt2e.quantizer import x86_inductor_quantizer as xiq
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
        from torchao.quantization.pt2e import move_exported_model_to_eval
    except ImportError as error:
        raise RuntimeError(
            "PT2E QAT requires torchao. Install the optional dependency with "
            "`pip install torchao`."
        ) from error
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        torchao_version = "unknown"
    if not torchao_version.startswith("0.17."):
        raise RuntimeError(
            f"This pipeline is validated with torchao 0.17.x, found {torchao_version}. "
            "Install `torchao==0.17.0`."
        )
    return prepare_qat_pt2e, convert_pt2e, X86InductorQuantizer, xiq, move_exported_model_to_eval


class BackboneBodyRegion(nn.Module):
    """Tensor-only export boundary returning C2-C5 as a stable tuple."""

    def __init__(self, body, feature_indices):
        super().__init__()
        self.body = body
        self.feature_indices = tuple(int(value) for value in feature_indices)

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in self.feature_indices:
                outputs.append(x)
        return tuple(outputs)


class _PT2EResNetBottleneck(nn.Module):
    """ResNet bottleneck with distinct ReLU module sources for PT2E matching.

    torchvision's Bottleneck invokes one ``self.relu`` module three times.
    TorchAO 0.17 groups those calls into one source partition with multiple
    outputs, while X86InductorQuantizer's QAT fusion matcher requires exactly
    one output per partition. Splitting the stateless ReLUs preserves the
    computation and state-dict names of all weighted layers.
    """

    def __init__(self, block):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.conv3 = block.conv3
        self.bn3 = block.bn3
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = block.downsample
        self.stride = block.stride

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu3(out)


def _split_resnet_bottleneck_relus(module):
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Bottleneck":
            setattr(module, name, _PT2EResNetBottleneck(child))
        else:
            _split_resnet_bottleneck_relus(child)
    return module


class ResNet50BodyRegion(nn.Module):
    """Explicit ResNet-50 export boundary with cloned C2-C5 outputs."""

    def __init__(self, body):
        super().__init__()
        self.stem = body[0]
        self.layer1 = _split_resnet_bottleneck_relus(body[1])
        self.layer2 = _split_resnet_bottleneck_relus(body[2])
        self.layer3 = _split_resnet_bottleneck_relus(body[3])
        self.layer4 = _split_resnet_bottleneck_relus(body[4])

    def forward(self, x):
        x = self.stem(x)
        # Put one clone immediately after every stage and feed that clone to
        # both the next stage and the returned feature tuple. Without this
        # boundary the last ReLU partition has two external output nodes
        # (the next stage and the detector output), which makes
        # X86InductorQuantizer QAT annotation fail for ResNet residual blocks.
        c2 = self.layer1(x).clone()
        c3 = self.layer2(c2).clone()
        c4 = self.layer3(c3).clone()
        c5 = self.layer4(c4).clone()
        return (c2, c3, c4, c5)


class BackboneBodyFPNRegion(nn.Module):
    """Optional second PT2E scope covering the backbone body and the complete FPN."""

    def __init__(self, body, fpn, feature_indices):
        super().__init__()
        self.body = BackboneBodyRegion(body, feature_indices)
        self.fpn = fpn

    def forward(self, x):
        tensors = self.body(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return tuple(self.fpn(features).values())


class ResNet50BodyFPNRegion(nn.Module):
    """Optional PT2E scope covering explicit ResNet-50 body and full FPN."""

    def __init__(self, body, fpn):
        super().__init__()
        self.body = ResNet50BodyRegion(body)
        self.fpn = fpn

    def forward(self, x):
        tensors = self.body(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return tuple(self.fpn(features).values())


def build_backbone_body_region(backbone, scope="backbone"):
    """Build the tensor-only PT2E export region for the selected backbone."""
    scope = str(scope).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("scope must be backbone or backbone_fpn")
    feature_indices = tuple(getattr(backbone, "feature_indices", (1, 3, 5, 7)))
    region_kind = str(getattr(backbone, "pt2e_region_kind", "convnext")).lower()
    if region_kind == "resnet50":
        return (
            ResNet50BodyRegion(backbone.body)
            if scope == "backbone"
            else ResNet50BodyFPNRegion(backbone.body, backbone.fpn)
        )
    return (
        BackboneBodyRegion(backbone.body, feature_indices)
        if scope == "backbone"
        else BackboneBodyFPNRegion(backbone.body, backbone.fpn, feature_indices)
    )


class PT2EBackboneFPN(nn.Module):
    """PT2E backbone graph followed by the original FP32 torchvision FPN."""

    def __init__(self, body_region, fpn, out_channels):
        super().__init__()
        self.body_region = body_region
        self.fpn = fpn
        self.out_channels = int(out_channels)

    def train(self, mode: bool = True):
        # Exported GraphModules reject .train()/.eval(). Their backbone graph is
        # captured in deterministic eval form; QAT fake-quant nodes still update
        # during forward. Only the ordinary FPN needs recursive mode switching.
        self.training = mode
        if self.fpn is not None:
            self.fpn.train(mode)
        return self

    def forward(self, x):
        tensors = self.body_region(x)
        features = OrderedDict((str(index), tensor) for index, tensor in enumerate(tensors))
        return self.fpn(features) if self.fpn is not None else features


def _dynamic_shapes(example_batch_size, maximum_batch_size, minimum_side, maximum_side,
                    spatial_divisor=32):
    # ConvNeXt downsamples by 32. Expressing this relation explicitly avoids
    # torch.export constraint violations while retaining variable image shapes.
    height = spatial_divisor * Dim(
        f"image_h_div_{spatial_divisor}", min=max(1, minimum_side // spatial_divisor),
        max=maximum_side // spatial_divisor,
    )
    width = spatial_divisor * Dim(
        f"image_w_div_{spatial_divisor}", min=max(1, minimum_side // spatial_divisor),
        max=maximum_side // spatial_divisor,
    )
    if maximum_batch_size > 1 and example_batch_size == 1:
        raise ValueError(
            "Dynamic batch export needs pt2e.example_batch_size >= 2; "
            "PyTorch specializes an example batch of one"
        )
    batch = Dim.STATIC if maximum_batch_size == 1 else Dim(
        "image_batch", min=1, max=maximum_batch_size,
    )
    return ({0: batch, 2: height, 3: width},)


def prepare_pt2e_backbone_qat(model, config, inplace=False):
    """Replace only backbone body with a prepared x86 PT2E QAT graph."""
    prepared_model = model if inplace else copy.deepcopy(model)
    if isinstance(prepared_model.backbone, PT2EBackboneFPN):
        return prepared_model
    pt2e = config.get("quantization", {}).get("pt2e", {})
    scope = str(pt2e.get("region", "backbone")).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("quantization.pt2e.region must be backbone or backbone_fpn")
    example_batch = int(pt2e.get("example_batch_size", 1))
    maximum_batch = int(pt2e.get("maximum_batch_size", config["training"].get("qat_batch_size", 1)))
    minimum_side = int(pt2e.get("minimum_image_side", 256))
    maximum_side = int(pt2e.get("maximum_image_side", config["model"].get("max_size", 1600)))
    example_height = int(pt2e.get("example_height", min(960, maximum_side)))
    example_width = int(pt2e.get("example_width", min(1280, maximum_side)))
    backbone = prepared_model.backbone
    region_kind = str(getattr(backbone, "pt2e_region_kind", "convnext")).lower()
    default_spatial_divisor = int(getattr(backbone, "pt2e_spatial_divisor", 32))
    spatial_divisor = 64 if scope == "backbone_fpn" else default_spatial_divisor
    if any(
        value <= 0 or value % spatial_divisor
        for value in (minimum_side, maximum_side, example_height, example_width)
    ):
        raise ValueError(
            f"PT2E {scope} image dimensions must be positive multiples of {spatial_divisor}"
        )
    if maximum_batch < example_batch:
        raise ValueError("pt2e.maximum_batch_size must be >= example_batch_size")

    region = build_backbone_body_region(backbone, scope=scope).cpu().eval()
    example = torch.randn(example_batch, 3, example_height, example_width)
    exported = export(
        region,
        (example,),
        dynamic_shapes=_dynamic_shapes(
            example_batch, maximum_batch, minimum_side, maximum_side,
            spatial_divisor,
        ),
    ).module()
    prepare_qat_pt2e, _, quantizer_type, xiq, _ = _torchao_pt2e()
    quantizer = quantizer_type()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=True))
    prepared_region = prepare_qat_pt2e(exported, quantizer)
    fake_quantizers = _collect_pt2e_fake_quantizers(prepared_region)
    if not fake_quantizers:
        raise RuntimeError(
            "prepare_qat_pt2e completed but inserted no PT2E fake-quant modules. "
            "This almost always means the active torch/torchao stack is incompatible. "
            f"torch={torch.__version__}. "
            "On Kaggle, reinstall a PT2E-compatible torch first, then rerun this notebook."
        )
    prepared_model.backbone = PT2EBackboneFPN(
        prepared_region, backbone.fpn if scope == "backbone" else None, backbone.out_channels,
    )
    prepared_model.pt2e_quantized_region = (
        "backbone.body" if scope == "backbone" else "backbone.body+fpn"
    )
    prepared_model.pt2e_backbone_kind = region_kind
    prepared_model.pt2e_export_spec = {
        "example_batch_size": example_batch,
        "maximum_batch_size": maximum_batch,
        "minimum_side": minimum_side,
        "maximum_side": maximum_side,
        "example_height": example_height,
        "example_width": example_width,
        "spatial_divisor": spatial_divisor,
    }
    if scope == "backbone_fpn":
        # FPN creates additional parity guards. Faster R-CNN must therefore pad
        # its internal ImageList to 64 instead of the default backbone stride.
        prepared_model.transform.size_divisible = 64
    return prepared_model


def set_pt2e_qat_phase(model, phase):
    """Set observer-only warmup, full fake-quant QAT, or frozen ranges."""
    if phase not in {"observer_warmup", "full", "frozen"}:
        raise ValueError("PT2E phase must be observer_warmup, full, or frozen")
    fake_quantizers = _collect_pt2e_fake_quantizers(model)
    if not fake_quantizers:
        raise RuntimeError(
            "No PT2E fake-quant modules found on the prepared model. "
            "The PT2E graph was likely created without QAT inserts because torch/torchao "
            "are incompatible in this runtime."
        )
    for module in fake_quantizers:
        if phase == "observer_warmup":
            module.enable_observer()
            module.disable_fake_quant()
        elif phase == "full":
            module.enable_observer()
            module.enable_fake_quant()
        else:
            module.disable_observer()
            module.enable_fake_quant()
    return len(fake_quantizers)


@contextmanager
def pt2e_observers_disabled(model):
    """Temporarily freeze PT2E observers during validation/benchmark inference."""
    fake_quantizers = _collect_pt2e_fake_quantizers(model)
    states = []
    for module in fake_quantizers:
        enabled = getattr(module, "observer_enabled", None)
        states.append(enabled.detach().clone() if torch.is_tensor(enabled) else None)
        module.disable_observer()
    try:
        yield len(fake_quantizers)
    finally:
        for module, enabled in zip(fake_quantizers, states):
            if enabled is not None:
                module.observer_enabled.resize_(enabled.shape).copy_(enabled)


def synchronize_pt2e_observers(model):
    """Merge min/max observer ranges across DDP ranks before rank-0 validation/save."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    synchronized = 0
    for module in _collect_pt2e_fake_quantizers(model):
        observer = getattr(module, "activation_post_process", None)
        minimum = getattr(observer, "min_val", None)
        maximum = getattr(observer, "max_val", None)
        if not (torch.is_tensor(minimum) and torch.is_tensor(maximum)):
            continue
        if minimum.numel() == 0 or maximum.numel() == 0:
            continue
        torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        if hasattr(module, "calculate_qparams"):
            scale, zero_point = module.calculate_qparams()
            module.scale.resize_(scale.shape).copy_(scale)
            module.zero_point.resize_(zero_point.shape).copy_(zero_point)
        synchronized += 1
    return synchronized


def inspect_pt2e_graph(model):
    """Return graph-level Q/DQ counts so tests can verify real PT2E conversion."""
    backbone = getattr(model, "backbone", None)
    region = getattr(backbone, "body_region", None)
    graph = getattr(region, "graph", None)
    if graph is None:
        return {"nodes": 0, "quantize": 0, "dequantize": 0, "quantized_ops": 0}
    targets = [str(node.target) for node in graph.nodes]
    return {
        "nodes": len(targets),
        "quantize": sum(
            "quantize_per_tensor" in target or "quantize_per_channel" in target
            for target in targets
        ),
        "dequantize": sum(
            "dequantize_per_tensor" in target or "dequantize_per_channel" in target
            for target in targets
        ),
        "quantized_ops": sum(
            "quantized_decomposed" in target
            and "dequantize" not in target
            and "quantize" not in target
            for target in targets
        ),
    }


def _model_signature(model):
    anchor_generator = model.rpn.anchor_generator
    box_predictor = model.roi_heads.box_predictor
    return {
        "region": getattr(model, "pt2e_quantized_region", None),
        "backbone_kind": getattr(model, "pt2e_backbone_kind", None),
        "num_classes": int(box_predictor.cls_score.out_features),
        "anchor_sizes": [list(map(float, sizes)) for sizes in anchor_generator.sizes],
        "aspect_ratios": [list(map(float, ratios)) for ratios in anchor_generator.aspect_ratios],
        "transform_min_size": list(map(int, model.transform.min_size)),
        "transform_max_size": int(model.transform.max_size),
        "transform_size_divisible": int(model.transform.size_divisible),
        "fpn_out_channels": int(model.backbone.out_channels),
    }


def convert_pt2e_backbone(model, inplace=False, compile_region=False):
    """Convert the prepared backbone graph and optionally compile that graph."""
    converted_model = model if inplace else copy.deepcopy(model)
    if not isinstance(converted_model.backbone, PT2EBackboneFPN):
        raise TypeError("Model has not been prepared by prepare_pt2e_backbone_qat")
    _, convert_pt2e, _, _, move_to_eval = _torchao_pt2e()
    converted_model.cpu()
    converted_region = convert_pt2e(converted_model.backbone.body_region)
    move_to_eval(converted_region)
    converted_model.backbone.body_region = converted_region
    converted_model.eval()
    converted_model.pt2e_compiled = False
    if compile_region:
        compile_pt2e_region(converted_model)
    return converted_model


def compile_pt2e_region(model):
    """Compile only the converted tensor graph, leaving detection control flow eager."""
    python_header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if not python_header.is_file():
        raise RuntimeError(
            f"torch.compile x86 requires Python development headers; missing {python_header}. "
            "Install python3-dev (or use a Kaggle image that provides Python.h)."
        )
    model.backbone.body_region = torch.compile(model.backbone.body_region)
    model.pt2e_compiled = True
    return model


def save_pt2e_int8_artifact(path, model, metrics=None, extra=None):
    """Persist the converted ExportedProgram plus the FP32 detector remainder.

    A plain state_dict is insufficient for PT2E: convert_pt2e embeds activation
    scales/zero-points as FX graph constants. Reconstructing an empty graph and
    loading only tensors silently changes predictions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = inspect_pt2e_graph(model)
    if graph["quantize"] == 0 or graph["dequantize"] == 0:
        raise RuntimeError("Refusing to save PT2E artifact: converted graph contains no Q/DQ nodes")
    spec = getattr(model, "pt2e_export_spec", None)
    if not isinstance(spec, dict):
        raise RuntimeError("PT2E export spec is missing; prepare and convert the model first")
    example = torch.randn(
        int(spec["example_batch_size"]), 3,
        int(spec["example_height"]), int(spec["example_width"]),
    )
    exported_program = export(
        model.backbone.body_region,
        (example,),
        dynamic_shapes=_dynamic_shapes(
            int(spec["example_batch_size"]), int(spec["maximum_batch_size"]),
            int(spec["minimum_side"]), int(spec["maximum_side"]),
            int(spec["spatial_divisor"]),
        ),
    )
    exported_buffer = io.BytesIO()
    torch.export.save(exported_program, exported_buffer)

    # Catch serialization regressions before a long benchmark discovers that
    # graph constants were dropped.
    reloaded_region = torch.export.load(io.BytesIO(exported_buffer.getvalue())).module()
    with torch.inference_mode():
        reference_features = model.backbone.body_region(example)
        reloaded_features = reloaded_region(example)
    if len(reference_features) != len(reloaded_features):
        raise RuntimeError("PT2E exported-region self-check changed the number of feature maps")
    for index, (reference, reloaded) in enumerate(zip(reference_features, reloaded_features)):
        try:
            torch.testing.assert_close(reference, reloaded, rtol=0, atol=1e-6)
        except AssertionError as error:
            raise RuntimeError(
                f"PT2E exported-region self-check failed for feature map {index}"
            ) from error

    detector_state = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.body_region.")
    }
    torch.save({
        "exported_region": exported_buffer.getvalue(),
        "detector_state": detector_state,
        "metrics": metrics or {},
        "extra": {
            **(extra or {}),
            "format": "pt2e_int8_exported_region",
            "format_version": 3,
            "model_signature": _model_signature(model),
            "graph": graph,
            "export_spec": spec,
            "torch_version": torch.__version__,
            "artifact_self_check": "passed",
        },
    }, path)
    return path


def load_pt2e_int8_artifact(path, config, compile_region=False):
    from ..models import build_fasterrcnn_convnext

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PT2E INT8 artifact not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Malformed PT2E INT8 artifact: expected a dictionary")
    artifact_format = payload.get("extra", {}).get("format")
    if artifact_format == "pt2e_int8_state_dict":
        raise ValueError(
            "Legacy PT2E state-dict artifact is incomplete because it does not preserve "
            "FX graph qparam constants. Regenerate it from pt2e_qat_best.pt with the "
            "current save_pt2e_int8_artifact()."
        )
    if artifact_format != "pt2e_int8_exported_region":
        raise ValueError("Not a PT2E INT8 exported-region artifact")
    if not isinstance(payload.get("exported_region"), bytes):
        raise ValueError("Malformed PT2E INT8 artifact: missing exported_region bytes")
    if not isinstance(payload.get("detector_state"), dict):
        raise ValueError("Malformed PT2E INT8 artifact: missing detector_state")
    saved_torch = payload.get("extra", {}).get("torch_version")
    if saved_torch and _normalized_version(saved_torch)[:2] != _normalized_version(torch.__version__)[:2]:
        raise RuntimeError(
            "PT2E ExportedProgram must be loaded with the same PyTorch major/minor version. "
            f"artifact={saved_torch}, runtime={torch.__version__}"
        )

    base_model = build_fasterrcnn_convnext(config)
    original_backbone = base_model.backbone
    active_backbone_kind = str(getattr(original_backbone, "pt2e_region_kind", "convnext")).lower()
    saved_signature = payload.get("extra", {}).get("model_signature")
    if not isinstance(saved_signature, dict):
        raise ValueError("Malformed PT2E INT8 artifact: missing model_signature")
    saved_region = saved_signature.get("region")
    saved_backbone_kind = saved_signature.get("backbone_kind")
    if saved_region not in {"backbone.body", "backbone.body+fpn"}:
        raise ValueError(f"Unsupported PT2E artifact region: {saved_region}")
    if saved_backbone_kind != active_backbone_kind:
        raise ValueError(
            "PT2E artifact backbone does not match the active config: "
            f"artifact={saved_backbone_kind}, config={active_backbone_kind}"
        )

    converted_region = torch.export.load(io.BytesIO(payload["exported_region"])).module()
    base_model.backbone = PT2EBackboneFPN(
        converted_region,
        original_backbone.fpn if saved_region == "backbone.body" else None,
        original_backbone.out_channels,
    )
    base_model.pt2e_quantized_region = saved_region
    base_model.pt2e_backbone_kind = active_backbone_kind
    base_model.pt2e_export_spec = payload.get("extra", {}).get("export_spec", {})
    if saved_region == "backbone.body+fpn":
        base_model.transform.size_divisible = 64
    model = base_model
    if saved_signature != _model_signature(model):
        raise ValueError(
            "PT2E artifact is incompatible with the active config. "
            f"saved_signature={saved_signature}, active_signature={_model_signature(model)}"
        )
    incompatible = model.load_state_dict(payload["detector_state"], strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected PT2E detector state keys: {incompatible.unexpected_keys}")
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith("backbone.body_region.")
    ]
    if invalid_missing:
        raise RuntimeError(f"Missing PT2E detector state keys: {invalid_missing}")
    model.cpu().eval()
    graph = inspect_pt2e_graph(model)
    if graph["quantize"] == 0 or graph["dequantize"] == 0:
        raise RuntimeError("Reloaded PT2E artifact has no Q/DQ nodes")
    if compile_region:
        compile_pt2e_region(model)
    return model, payload
