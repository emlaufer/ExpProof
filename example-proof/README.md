# Example Proof

This directory contains a sample ExpProof proof.
It contains the following files:

- `network.onnx`: An onnx file containing the model
- `input.json`: The input point file
- `model.compiled`: The compiled model by EZKL
- `settings.json`: The EZKL settings file
- `witness.json`: The expanded witness
- `vk.key`: The verifying key
- `proof.json`: The inferrence and explanation proof

You can verify the proof by running `ezkl verify`
