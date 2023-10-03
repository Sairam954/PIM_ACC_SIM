from os.path import isfile, join
from os import listdir, name
import pandas as pd
import torch 
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
from config import *

cnnModelDirectory = "CNNModels//"
modelList = [f for f in listdir(cnnModelDirectory) if isfile(join(cnnModelDirectory, f))]
modelList = ['GoogLeNet.csv']

ns_to_sec = 1e-9
us_to_sec = 1e-6

nJ_to_J = 1e-9
mW_to_W = 1e-3

accelerator_list = [DRISA_1T1C_NOR]

for tpc in accelerator_list:
    architecture = tpc[0][NAME]
    batch_size = tpc[0][BATCH_SIZE]
    dataflow = tpc[0][DATAFLOW]
    pe = tpc[0][PE]
    mul_cycle = tpc[0][MUL_CYCLE]
    add_cycle = tpc[0][ADD_CYCLE]
    data_move_latency = tpc[0][DATA_MOVE_LATENCY]
    mac_energy = tpc[0][MAC_ENERGY]
    tRc = tpc[0][TRC]
    accelerator_peformance = []
    print("Architecture", architecture, "Dataflow ", dataflow)
    for modelName in modelList:
        total_data_movement_latency = 0
        total_latency = 0
        total_mac_energy = 0
        utilized_PE = 0
        idle_PE = 0   
        result = {}  
        print("Model being Processed ", modelName)
        nnModel = pd.read_csv(cnnModelDirectory+modelName)
        nnModel = nnModel.astype({"model_name": str, 'name': str, 'kernel_depth': int, 'kernel_height': int, 'kernel_width': int,	'tensor_count': int, 'input_shape': str,
                             'output_shape': str, 'tensor_shape': str,	'input_height': int,	'input_width': int, 'input_depth': int, 'output_height': int, 'output_width': int, 'output_depth': int})
        nnModel = pd.concat([nnModel]*batch_size, ignore_index=True)
        accelerator_model_inference = {}
        for idx in nnModel.index:
            layer_type = nnModel[LAYER_TYPE][idx]
            model_name = nnModel[MODEL_NAME][idx]
            kernel_depth = nnModel[KERNEL_DEPTH][idx]
            kernel_width = nnModel[KERNEL_WIDTH][idx]
            kernel_height = nnModel[KERNEL_HEIGHT][idx]
            tensor_count = nnModel[TENSOR_COUNT][idx]
            input_shape = nnModel[INPUT_SHAPE][idx]
            output_shape = nnModel[OUTPUT_SHAPE][idx]
            tensor_shape = nnModel[TENSOR_SHAPE][idx]
            input_height = nnModel[INPUT_HEIGHT][idx]
            input_width = nnModel[INPUT_WIDTH][idx]
            input_depth = nnModel[INPUT_DEPTH][idx]
            output_height = nnModel[OUTPUT_HEIGHT][idx]
            output_width = nnModel[OUTPUT_WIDTH][idx]
            output_depth = nnModel[OUTPUT_DEPTH][idx]
            stride_height = 1
            stride_width = 1
            print('Layer', layer_type)
            in_channels = kernel_depth
            out_channels = kernel_height * kernel_width * in_channels
            out_height = (input_height - kernel_height) // stride_height + 1
            out_width = (input_width - kernel_width) // stride_width + 1
            inp = torch.randn(batch_size, in_channels, input_height, input_width)
            w = torch.randn(tensor_count, kernel_depth, kernel_height, kernel_width)
            
            # Tranformation of convolutions and Fully connected layer operations into GEMM
            if layer_type=='Conv2D' or layer_type=='PointWiseConv': 
                toeplitz_input = torch.nn.functional.unfold(inp, kernel_size=(kernel_height, kernel_width), stride=(stride_height, stride_width))
                toeplitz_input = toeplitz_input.view(out_channels*batch_size, out_height*out_width)
                toeplitz_w = w.view(w.size(0), -1)
            elif layer_type=='Dense':
                toeplitz_input = inp.flatten().view(-1, 1)
                toeplitz_w = w.flatten().view(1, -1)
            toeplitz_w = torch.transpose(toeplitz_w, 0 , 1)
            # output = toeplitz_w @ toeplitz_input
            D = toeplitz_w.shape[1]
            C = toeplitz_input.shape[0]
            K = toeplitz_input.shape[1]
            # D = 100
            # C = 100
            # K = 100
            M = pe # Number of PE
            I = torch.randn(C,K)
            W = torch.randn(K,D)
            O = torch.zeros(C,D)
            # print('Layer', layer_type)
            for d in range(0,D):
                for k in range(0, K):
                    w_slice = W[k, d] # weight stationary same weight is used by all the PEs 
                    for c in range(0,C,M):
                        i_slice = I[c: min(c+M,C), k] # Each PE gets a different input value where each input corresponds to a different output pixel
                        i_slice = i_slice.reshape(-1, 1)
                        dpu_w_slice = w_slice
                        dpu_w_slice = dpu_w_slice.T.repeat(min(c+M,C)-c,1)
                        #  print(dpu_w_slice.shape)
                        #  print(i_slice.shape)
                        psum_dpu = torch.einsum('ij,ij->i', i_slice, dpu_w_slice)
                        O[c:c+M,d] = psum_dpu+O[c:c+M,d]
                        utilized_PE += min(C,M) 
                        idle_PE += M-min(C,M)
                        total_latency += (mul_cycle+add_cycle)*tRc+data_move_latency
                        total_data_movement_latency += data_move_latency
                        total_mac_energy = min(C,M)*mac_energy
                    # completed number of multiplications = number of PEs or (M-C)      
            # print(O)
            # print(I @ W)
            # print(torch.allclose(O, I @W))
        print('Total Latency', total_latency)
        print('Total Data Movement', total_data_movement_latency)
        print('Total MAC energy', total_mac_energy)
        print('Total Utilized PEs', utilized_PE)
        print('Total Idle PEs', idle_PE)
        # print('Utilized to Idle PE Ratio', utilized_PE/idle_PE)
        
        
        accelerator_model_inference = {"Name":architecture,"Model":model_name,"Total_Run_Time": total_latency,"Total_Data_Movement": total_data_movement_latency,"Total_MAC_Energy": total_mac_energy,"Utilized_PE": utilized_PE,"Idle_PE": idle_PE, "BottleNeck_Ratio": total_data_movement_latency/total_latency}
        accelerator_peformance.append(accelerator_model_inference)
    accelerator_peformance_df = pd.DataFrame(accelerator_peformance)
    accelerator_peformance_df.to_csv('acclerator_performance.csv',index=False)