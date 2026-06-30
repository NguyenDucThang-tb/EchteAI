import onnx
import onnxruntime as ort
import torch
import numpy as np
import re
import os
import torch.nn.functional as F
import logging

def onnx_conv_outputs_from_batch(
    model_path,
    images,
    pattern=r".*conv.*",
    transform=None,
    device=None,
    layer=None,
    num_layers=None,
    last_n_layers=None,
    print_layers=False
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device).lower()

    if transform is not None:
        img_list, _ = transform(images)
        input_tensor = img_list.tensors.to(device)
    else:
        if isinstance(images, list):
            input_tensor = torch.stack(images).to(device)
        else:
            input_tensor = images.to(device)

    model = onnx.load(model_path)

    conv_outputs = []
    for node in model.graph.node:
        if node.op_type.lower() == "conv":
            for output in node.output:
                if re.search(pattern, output, re.IGNORECASE):
                    conv_outputs.append(output)

    if len(conv_outputs) == 0:
        raise RuntimeError("No convolution outputs matched pattern")

    if print_layers:
        print("Matched conv layers:")
        for i, name in enumerate(conv_outputs):
            print(f"{i}: {name}")

    if layer is not None:
        if layer < 0 or layer >= len(conv_outputs):
            raise ValueError(f"layer index {layer} out of range (0-{len(conv_outputs)-1})")
        conv_outputs = [conv_outputs[layer]]
    elif num_layers is not None:
        conv_outputs = conv_outputs[:num_layers]
    elif last_n_layers is not None:
        conv_outputs = conv_outputs[-last_n_layers:]

    existing_outputs = [o.name for o in model.graph.output]
    from onnx import helper, TensorProto
    for name in conv_outputs:
        if name not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )

    export_path = model_path.replace(".onnx", "_with_outputs.onnx")
    onnx.save(model, export_path)

    providers = ['CUDAExecutionProvider'] if "cuda" in device else ['CPUExecutionProvider']
    session = ort.InferenceSession(export_path, providers=providers)
    input_name = session.get_inputs()[0].name

    ort_outputs = session.run(None, {input_name: input_tensor.cpu().numpy()})
    output_names = [o.name for o in session.get_outputs()]

    outputs = {
        name: torch.tensor(val, device=device)
        for name, val in zip(output_names, ort_outputs)
        if re.search(pattern, name, re.IGNORECASE)
    }

    return outputs
