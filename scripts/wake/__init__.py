"""Tools for MIST's wake model: the recording manifest check and the evaluation harness.

The recordings are personal audio that never enters the repository, so these tools read them
only from the path a manifest names, and write nothing but file ids, counts and rates.
"""
