NAME = "NAME"
PE = "PE"
TRC = "tRC"
MUL_CYCLE = "MUL_CYCLE"
ADD_CYCLE = "ADD_CYCLE"
DATA_MOVE_LATENCY = "DATA_MOVE_LATENCY"
DATAFLOW = "DATAFLOW"
MAC_ENERGY = "MAC_ENERGY"
BATCH_SIZE = "BATCH_SIZE"
LAYER_TYPE = "name"
MODEL_NAME = "model_name"
KERNEL_DEPTH = "kernel_depth"
KERNEL_HEIGHT = "kernel_height"
KERNEL_WIDTH = "kernel_width"
TENSOR_COUNT = "tensor_count"
INPUT_SHAPE = "input_shape"
OUTPUT_SHAPE = "output_shape"
TENSOR_SHAPE = "tensor_shape"
INPUT_HEIGHT = "input_height"
INPUT_WIDTH = "input_width"
INPUT_DEPTH = "input_depth"
OUTPUT_HEIGHT = "output_height"
OUTPUT_WIDTH = "output_width"
OUTPUT_DEPTH = "output_depth"

DRISA_1T1C_NOR =    [{
    NAME: "DRISA_1T1C_NOR",
    PE: 32768,    
    TRC: 8, # ns
    MUL_CYCLE: 200,
    ADD_CYCLE: 21,
    DATA_MOVE_LATENCY: 8, #ns
    DATAFLOW: "WS", 
    MAC_ENERGY: 6630, # pJ
    BATCH_SIZE: 1
}]