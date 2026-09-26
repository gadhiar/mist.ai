"""Stateless HTTP extraction service (T1a).

Runs Stage 1 pre-processing, Stage 1.5 scope classification, Stage 2
ontology extraction, and Stage 9 internal derivation as a separate
process/host from the MIST.AI backend, so extraction can run on its own
llama-server and model without the backend's Neo4j/event-store/graph-writer
dependencies. See `backend/extraction_contract/models.py` for the wire
contract this service and the backend dispatcher (T2) share.
"""
