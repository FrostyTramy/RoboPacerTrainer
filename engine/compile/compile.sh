#!/bin/bash
# Usage: bash compile.sh <model_name>
set -e

NET_NAME=${1:-model}
MODELS_DIR=/workspace/models
ONNX_MODEL=$MODELS_DIR/${NET_NAME}.onnx
HAR_FILE=$MODELS_DIR/${NET_NAME}.har
CALIB_NCHW=$MODELS_DIR/calib_data.npy
CALIB_NHWC=$MODELS_DIR/calib_data_nhwc.npy
OPT_HAR=$MODELS_DIR/${NET_NAME}_optimized.har
HW_ARCH=hailo8

echo "============================================================"
echo "Hailo DFC  model=${NET_NAME}  target=${HW_ARCH}"
echo "============================================================"

# Fix: Ubuntu pip uses dist-packages but DFC checks for site-packages
python3 << 'PYEOF'
import os, glob
py_path = '/usr/local/lib/python3.10/dist-packages/hailo_sdk_common/paths_manager/paths.py'
with open(py_path, 'r') as f:
    src = f.read()
OLD = '"site-packages" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin))'
NEW = ('("site-packages" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin))'
       ' or "dist-packages" in os.path.dirname(os.path.dirname(hailo_sdk_common.origin)))')
if OLD in src:
    src = src.replace(OLD, NEW)
    with open(py_path, 'w') as f: f.write(src)
    for pyc in glob.glob(os.path.join(os.path.dirname(py_path), '__pycache__', 'paths.cpython-*.pyc')):
        os.remove(pyc)
    print('DFC patched')
PYEOF

# Convert calibration data NCHW -> NHWC
python3 -c "
import numpy as np
d = np.load('$CALIB_NCHW')
print(f'calib: {d.shape} -> ', end='')
nhwc = np.transpose(d,(0,2,3,1)).astype(np.float32)
np.save('$CALIB_NHWC', nhwc)
print(nhwc.shape)
"

echo ""
echo "[ 1/3 ] Parsing ONNX -> HAR ..."
hailo parser onnx "$ONNX_MODEL" \
    --hw-arch "$HW_ARCH" --net-name "$NET_NAME" --har-path "$HAR_FILE" -y

echo "[ 2/3 ] Optimizing (PTQ int8) ..."
hailo optimize "$HAR_FILE" \
    --hw-arch "$HW_ARCH" --calib-set-path "$CALIB_NHWC" --output-har-path "$OPT_HAR"

echo "[ 3/3 ] Compiling -> HEF ..."
cd "$MODELS_DIR"
hailo compiler "$OPT_HAR" --hw-arch "$HW_ARCH" --output-dir "$MODELS_DIR"

echo ""
echo "SUCCESS -> $MODELS_DIR/${NET_NAME}.hef"
