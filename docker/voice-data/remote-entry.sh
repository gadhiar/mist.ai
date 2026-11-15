#!/bin/sh
# Default command of every stage image on a rented pod. When the pod sets PUBLIC_KEY, start sshd
# with key-only login for that key; then wait, so the pod stays up for `remote` jobs. No key is
# baked into the image: host keys are generated here, per pod, and the login key comes from the
# pod's environment. A local `docker run` names its own command and never reaches this script.
set -eu

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    printf '%s\n' \
        'PasswordAuthentication no' \
        'KbdInteractiveAuthentication no' \
        'PermitRootLogin prohibit-password' \
        > /etc/ssh/sshd_config.d/mist-voice.conf
    ssh-keygen -A
    /usr/sbin/sshd
fi

exec sleep infinity
