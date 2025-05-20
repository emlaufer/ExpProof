<h1 align="center">
ExpProof
</h1>

This is a fork of [EZKL](https://github.com/zkonduit/ezkl) which supports zero-knowledge proof of explanations,
used for our [research paper](https://arxiv.org/abs/2502.03773). 

You can find a tutorial in [tutorial.md](tutorial.md).

# ExpProof Modification Locations

For the main circuit implementation which checks the generated explanation,
go to [`src/circuit/modules/lime.rs`](src/circuit/modules/lime.rs).

For functions related to witness generation (i.e. running lime),
go to [`src/graph/lime.rs`](src/graph/lime.rs).

To configure ablations for benchmarking, go to [`src/ablate.rs`](src/graph/lime.rs)

# Example Proof

You can find an example proof in the [example-proofs](example-proofs) directory.
