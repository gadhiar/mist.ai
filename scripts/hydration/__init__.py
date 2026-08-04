"""R1.4.6 hydration tooling.

`manifest` records what produced a hydration artifact; `snapshot` dumps the
dev stack into one and restores it back. The hydrator itself (T2) drives the
authored corpus through the live path and is separate work -- this package is
the container it runs in and the way its expensive output survives.
"""
