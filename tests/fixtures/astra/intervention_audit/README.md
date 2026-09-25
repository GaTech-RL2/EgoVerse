# Small audit fixture

`arrays.npz` is synthetic CPU data, not an experiment recording. It contains one
seeded float32 flow state of shape `[1, 10, 32]`, eight orthogonal unit-RMS basis
vectors, a zero full action endpoint, and a constant decoded seven-channel action
endpoint. It has no images, provider responses, credentials, or model weights.

The known noise uses NumPy's default generator with
`SeedSequence([19, int(digest("fixture:seed19:task0:state0")[:8], 16)])`.
The first eight coordinate vectors, multiplied by `sqrt(320)`, form the basis.
The tests mutate recorded bytes, bind a valid new array to the wrong latent, and
change otherwise internally consistent reset records. They never tokenize, solve
a model, download data, or start a simulator.
