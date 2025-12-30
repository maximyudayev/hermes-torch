# HERMES - PyTorch

Support package to inject [PyTorch](https://pytorch.org/) AI models in the sensing-processing closed-loop of [HERMES](https://github.com/maximyudayev/hermes).

> [!NOTE]
> In principle, any PyTorch model that runs in your existing AI workflow (e.g. on your laptop, single-board computer, server, edge device, etc.) is runnable via this wrapper.

- [x] Classifier model wrapper, via `TorchClassifierPipeline`.
- [ ] Regression model wrapper.
- [ ] ... 

## Installation
Nodes available under the same HERMES namespace of `hermes.torch`, such as `TorchClassifierPipeline`.

> [!NOTE]
> On Windows, [sometimes Visual Studio Redistributable and Runtime](https://github.com/pytorch/pytorch/issues/166628#issuecomment-3479375122) are not installed, which would throw dynamic library load errors when importing `torch`. Make sure to download it if you have a clean system and are running into issues importing PyTorch on your system.

### From PyPI
```bash
pip install pysio-hermes-torch
```

### From source
```bash
git clone https://github.com/maximyudayev/hermes-torch.git
pip install -e hermes-torch
```

## Usage
Using PyTorch AI models follows the standard [configuration file specification](https://yudayev.com/hermes) process of HERMES nodes.

1. Prepare your PyTorch model in a regular workflow - design, train, and export a `.pth` checkpoint.
1. Provide the path in the HERMES config file to both, the checkpoint file and the module containing the `nn.Module` architecture.
1. Specify the input modalities the AI model should receive for its computations under [`stream_in_specs`](https://github.com/maximyudayev/hermes-torch/blob/main/examples/torch.yml#L101-L122).

> [!IMPORTANT]
> Ensure that the state dictionary of the `.pth` can be successfully loaded into the model: provide the same model hyperparameters as the ones used to construct the model for training under [`stream_out_spec.module_params`](https://github.com/maximyudayev/hermes-torch/blob/main/examples/torch.yml#L92-L99).

> [!IMPORTANT]
> HERMES will push data from the middleware to the model input in `process_data` in an event-driven way. Samples from one or more modalities may be available in each iteration of the call. It is currently the responsibility of the user to buffer or fuse the incoming multi-modal samples within the `nn.Module` to build up receptive fields of desired function. The `process_data` guarantees that the per-modality `Queue` connected to the model's input has the most recent sample only.

### (Optional) Dedicated `Node`
You can optionally inherit from `TorchClassifierPipeline`, or `Pipeline` directly, to create a dedicated (hardcoded) `Node` for the target AI algorithm, similar to the HERMES extension process with support for new sensors (e.g. a `TorchRegressionPipeline`, `TorchEmbeddingPipeline`, etc., some of which are on the current development roadmap).
This will simplify the YAML file structure, especially for big models.
In that case, make sure to override the required abstract methods to enable auto-discovery and connection of the custom node to the rest of HERMES.

## Citation
When using any parts of this repository outside of its intended use, please cite the parent project [HERMES](https://github.com/maximyudayev/hermes).
