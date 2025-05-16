<h1 align="center">
ExpProof
</h1>

This is a fork of ![EZKL](https://github.com/zkonduit/ezkl) which supports zero-knowledge proof of explanations,
used for our ![research paper](https://arxiv.org/abs/2502.03773). 

# Getting Started with Zero-Knowledge Explanations

Running ExpProof is as simple as running ezkl. We currently only support the command line interface.

First, you will need to have exported your model as an ONNX file.

## Generate Settings

Run the following command to generate the settings.json file:
`ezkl gen-settings`

Here is an example with the types of settings you will need for explanations:

`ezkl gen-settings --generate-explanation 100 --top-k 10 --lime-sampling gaussian --lime-weight-strategy exponential --lookup-range='-100000->100000' --logrows 18 --model network.onnx`

This command generates settings for Lime explanations with 100 sample points, gives the top-10 LIME
coefficinents as the public explanation, uses gaussian sampling, and exponential weighting.
ExpProof doesn't properly determine the number of rows needed for the Halo2 table, so you
have to find it manually through trial and error based on your parameters and model.
The lookup range may need to be configured based on your input point.

Next, compile the circuit:

`ezkl compile-circuit -S settings.json --model network.onnx`

## Witness Generation

Generate the witness using:

`ezkl gen-witness -d input.json`

This contains the json encoding for the input point. Due to some idiosyncracies in the code,
we require that the input json for a single point have three rows: the first is the input point, the second
is any 23 length vector, and the last is a vector containing a single field element.
See `example_input.json` as an example for a model with dimension 23.

## Setup circuit

Generate the public and private key using

`ezkl setup -W witness.json --vk-path=vk.key --pk-path=pk.key`

Due to idiosyncracies in the implementation, we require giving a valid witness
during setup (though it does not bind the proving and verifying key to the witness).

## Proving and Verification

Generate a proof using

`ezkl prove`

Verify it using

`ezkl verify`

# ExpProof Modification Locations

For the main circuit implementation which checks the generated explanation,
go to `src/circuit/modules/lime.rs`.

For functions related to witness generation (i.e. running lime),
go to `src/graph/lime.rs`.

To configure ablations for benchmarking, go to `src/ablate.rs`
