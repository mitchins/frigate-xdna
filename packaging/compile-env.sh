# Compiler environment for the product appliance (payload-relative).
# Sourced ONLY by the compiler launcher for the short-lived compile child.
# Never leaks into the manager or the native inference child.
export RYZEN_AI_INSTALLATION_PATH=/opt/compile-venv
SP=$RYZEN_AI_INSTALLATION_PATH/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$SP/flexml/flexml_extras/lib:$SP/onnxruntime/capi:$SP/voe/lib:$SP/lnx64.o/tools/peano/lib:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/opt/xilinx-xrt/lib:$LD_LIBRARY_PATH
unset PYTHONPATH
export XILINX_VITIS=$SP
export XILINX_VITIS_AIETOOLS=$SP
export XILINX_XRT=/opt/xilinx-xrt
