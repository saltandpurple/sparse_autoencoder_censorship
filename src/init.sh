#!/bin/bash

tailscaled --tun=userspace-networking --socks5-server=localhost:1055 >/tmp/ts.log 2>&1 &
# tailscale up

# todo: pull models
exec sleep infinity