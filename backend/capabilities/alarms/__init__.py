"""Native, offline alarms and timers.

`timeparse.py` reads a time as the user said it, because the model passes the user's words rather
than a computed timestamp, and computes the next occurrence of a repeat rule. `store.py` keeps
alarms durable in the shared capabilities database, so ringing and missed state survive a
restart and a client that connects late.

This package file holds no imports, as its parent package's does not.
"""
