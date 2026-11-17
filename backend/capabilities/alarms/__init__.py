"""Native, offline alarms and timers.

`timeparse.py` reads a time as the user said it, because the model passes the user's words rather
than a computed timestamp, and computes the next occurrence of a repeat rule. `records.py` keeps
each alarm or timer as a record in the shared capabilities database, with its live occurrences and
an event trail, so ringing and missed state survive a restart and a client that connects late, and
a daily alarm keeps one record however often it fires. `scheduler.py` is the one writer of
occurrence transitions, and `notifier.py` tells clients about them.

This package file holds no imports, as its parent package's does not.
"""
