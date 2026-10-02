#!/bin/bash
set -euo pipefail
exec > >(tee /var/log/user-data.log | logger -t startup-script -s 2>/dev/console) 2>&1

echo "Starting startup script for CPU LightGBM benchmark node"

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip libgomp1

# Keep lab packages separate from Debian's managed system Python.
python3 -m venv /opt/lab16-venv
/opt/lab16-venv/bin/python -m pip install --upgrade pip
/opt/lab16-venv/bin/python -m pip install lightgbm scikit-learn pandas numpy kaggle
printf '%s\n' 'export PATH="/opt/lab16-venv/bin:$PATH"' > /etc/profile.d/lab16.sh
/opt/lab16-venv/bin/python -c "import lightgbm, sklearn, pandas, numpy; print('ML imports OK')"

echo "CPU environment ready: lab packages installed in /opt/lab16-venv."
