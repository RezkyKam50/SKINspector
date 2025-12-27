#!/bin/bash

verified_count=0
CUDA_V=V12.9.86

echo "checking uv installation ..."
if command -v uv &>/dev/null; then
    echo "VERIFIED: uv is installed: $(uv --version)"
    ((verified_count++))
else
    echo "WARN: uv is NOT installed"
fi

echo
echo "checking Python sqlite3 support ..."
if python -c "import sqlite3; print('VERIFIED: sqlite3 available (linked SQLite version:', sqlite3.sqlite_version, ')')" 2>/dev/null; then
    ((verified_count++))
else
    echo "WARN: sqlite3 (or _sqlite3) not available in current Python"
fi

echo
echo "checking Python dev headers (Python.h) ..."
PYTHON_H_PATH=$(python -c "import sysconfig; print(sysconfig.get_paths().get('include', ''))" 2>/dev/null)
if [[ -n "$PYTHON_H_PATH" && -f "$PYTHON_H_PATH/Python.h" ]]; then
    echo "VERIFIED: Python.h found in $PYTHON_H_PATH"
    ((verified_count++))
else
    echo "WARN: Python.h not found (you may need pythonX.Y-dev / pythonX.Y-devel)"
fi

echo
echo "checking CUDA ${CUDA_V} ..."
if command -v nvcc &>/dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    if [[ "$cuda_version" == $CUDA_V ]]; then
        echo "VERIFIED: CUDA $cuda_version detected"
        ((verified_count++))
    else
        echo "WARN: CUDA detected but version is $cuda_version (expected ${CUDA_V})"
    fi
else
    echo "WARN: CUDA (nvcc) not found"
fi

echo
if [[ $verified_count -eq 4 ]]; then
    echo "All checks VERIFIED. Running build script ..."
    bash ./scripts/dependencies.sh
    echo "Setting up Linux desktop integration ..."
    chmod +x ./scripts/linux_desktop.sh
    bash ./scripts/linux_desktop.sh
    echo "Setup completed."
else
    echo "FATAL: Not all requirements verified."
fi


