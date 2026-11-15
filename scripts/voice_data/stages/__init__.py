"""Stage modules: the code a job runs inside its family's image.

Each runs as `<venv>/bin/python -I -m scripts.voice_data.stages.<name> --data /data ...` through
`jobs.LocalDocker`, may import third-party packages the image holds, and writes `outputs.json`
for the executor to verify. Host-side purity does not apply here.
"""
