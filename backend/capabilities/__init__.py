"""MIST's capability layer: tools that act on the world, such as lights, music and alarms.

Each capability declares typed tool contracts, a risk class, a confirmation policy and routing
hints, and runs behind a provider (`contracts.py`). Offering a turn only a few small tools is
what lets a small model choose among them reliably.

This package file holds no imports: modules import what they need from the defining module, so
adding a capability never edits this file.
"""
