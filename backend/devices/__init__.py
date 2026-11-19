"""Room devices: the backend side of a device that reaches MIST over the LAN.

- `registry.py` reads the registered devices, which `scripts/daily/devices.py` writes on the host.
- `listener.py` serves `/v2/device`: it authenticates a device before accept, then serves it one
  realtime session.
- `server.py` runs the listener as a second uvicorn server, with TLS, in the backend's process.
- `limiter.py` limits failed authentications by the device id a token claims.
- `config.py` reads the listener's `MIST_DEVICE_*` settings.
- `ring_sink.py` gives each registered, unrevoked device a ring endpoint for `alarm_fired`.

The package is apart from the desktop path because the listener is a second socket: it is off
unless `MIST_DEVICE_LISTENER=on`, and a listener that fails to start leaves the desktop path
running.

This package file holds no imports: modules import what they need from the defining module.
"""
