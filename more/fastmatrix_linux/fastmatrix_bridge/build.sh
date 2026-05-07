#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_FILE="$ROOT_DIR/cpp/fastmatrix.cpp"
OUT_DIR="$ROOT_DIR/fastmatrix"
UNAME="$(uname -s)"

if [ "$UNAME" = "Darwin" ]; then
  OUT_LIB="$OUT_DIR/libfastmatrix.dylib"
  g++ -O3 -std=c++17 -shared -fPIC -pthread "$CPP_FILE" -o "$OUT_LIB"
elif [[ "$UNAME" == MINGW* ]] || [[ "$UNAME" == MSYS* ]] || [[ "$UNAME" == CYGWIN* ]]; then
  OUT_LIB="$OUT_DIR/fastmatrix.dll"
  g++ -O3 -std=c++17 -shared -pthread "$CPP_FILE" -o "$OUT_LIB"
else
  OUT_LIB="$OUT_DIR/libfastmatrix.so"
  g++ -O3 -std=c++17 -shared -fPIC -pthread "$CPP_FILE" -o "$OUT_LIB"
fi

echo "Built: $OUT_LIB"
