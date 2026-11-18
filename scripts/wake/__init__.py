"""Tools for MIST's wake model: the guided recorder, the recording manifest check and the
evaluation harness.

The recordings are personal audio that never enters the repository. The recorder writes them
only under the `--root` it is given, outside the repository; the check and the harness read them
only from the path a manifest names, and write nothing but file ids, counts and rates.
"""
