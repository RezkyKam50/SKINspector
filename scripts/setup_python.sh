#!/bin/bash
set -e

sudo rm -rf .venv

source ./scripts/setup/cuda_toolkit.sh
source ./scripts/setup/gcc_switcher.sh

PYTHON_VERSION="3.13.7"
PY_ABI="3.13"
PYTHON_TAR="Python-${PYTHON_VERSION}.tar.xz"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TAR}"
INSTALL_PREFIX="/usr/local"  

download_file() {
    local url="$1"
    local output="$2"
    if command -v aria2c &> /dev/null; then
        echo "Using aria2c for faster download..."
        aria2c -s 16 -x 16 "$url" -o "$output"
    elif command -v wget &> /dev/null; then
        echo "aria2c not found, using wget..."
        wget "$url" -O "$output"
    elif command -v curl &> /dev/null; then
        echo "Neither aria2c nor wget found, using curl..."
        curl -L "$url" -o "$output"
    else
        echo "Error: No download tool found. Please install aria2c, wget, or curl."
        exit 1
    fi
}

if [ ! -f "$PYTHON_TAR" ]; then
    echo "Downloading Python ${PYTHON_VERSION}..."
    download_file "$PYTHON_URL" "$PYTHON_TAR"
else
    echo "Python tarball already exists, skipping download."
fi

if [ ! -s "$PYTHON_TAR" ]; then
    echo "Error: Downloaded file is empty or doesn't exist."
    exit 1
fi

echo "Extracting Python source..."
tar xvf "$PYTHON_TAR"

cd "Python-${PYTHON_VERSION}"

echo "Configuring Python build..."
./configure --prefix="$INSTALL_PREFIX" \
    --enable-optimizations \
    --with-lto \
    --enable-shared

echo "Building Python (this may take a while)..."
make -j$(nproc --all)

echo "Installing Python system-wide..."
sudo make altinstall

echo "Updating shared library cache..."
sudo ldconfig
 
echo "Configuring library path..."
if ! grep -q "/usr/local/lib" /etc/ld.so.conf.d/local.conf 2>/dev/null; then
    echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf
    sudo ldconfig
fi
 
export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

cd ..

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python${PY_ABI} -m venv .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Python ${PYTHON_VERSION} installation completed successfully!"

sudo rm -rf Python* && sudo rm -rf python*