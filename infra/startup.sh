#!/bin/bash
apt-get update -qq && apt-get install -y -qq libosmesa6
grep -q arena-venv /root/.bashrc || cat >> /root/.bashrc << 'RC'
source /workspace/arena-venv/bin/activate
cd /workspace/ARENA
RC
git config --global user.name "Blake Arnold"
git config --global user.email "blakecarnold7@gmail.com"
git config --global credential.helper 'store --file /workspace/.git-credentials'
