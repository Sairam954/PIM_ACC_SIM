# PIM_ACC_SIM
This is a simulator for evaluating the inference performance of In-DRAM CNN accelerators like DRISA, ATRIA, SCOPE, and LACC

### Installation and Execution

    git clone https://github.com/Sairam954/PIM_ACC_SIM.git
    python main.py

### Video Tutorial
https://youtu.be/CHnlewJt8-Q


### Accelerator Configuration 

The accelerator configuration can be provided in config.py file. The configuration dictionary looks like below:
``` bash
DRISA_1T1C_NOR =    [{
    NAME: "DRISA_1T1C_NOR", # Name of the accelerator, this name will used for logging performance results in accelerator_performance.csv file 
    PE: 32768,  # Number of Processing Elements in the accelerators
    TRC: 8, # ns 
    MUL_CYCLE: 200, # Latency to perform multiplication in ns
    ADD_CYCLE: 21, # Latency to perform Addition in ns
    DATA_MOVE_LATENCY: 8, # Latency corresponding to the data movement in ns
    DATAFLOW: "WS",  # Execution dataflow, currently framework can only map using Weight Stationary (WS) dataflow. However, it can be extended to map Input Stationary and Output Stationary dataflows 
    MAC_ENERGY: 6630, # Energy consumed to perform a MAC operation in pJ
    BATCH_SIZE: 1 # The input batch size 
}]
```

### PIM_ACC_SIM Project Structure 
``` bash
PIM_ACC_SIM
    ├── acclerator_performance.csv # Results of Simulations are stored in this file. Simulator captures metrics like Total_Run_Time, Total_Data_Movement,	Total_MAC_Energy, and	BottleNeck_Ratio 
    ├── Atria_PAPER_results_modified_V13.xlsx # Excel sheet containing the latency, energy, and area values corresponding to different accelerators, these values are used to create configuration dictionaries in config.py 
    ├── Config.py # File is used to declare the accelerator configurations used for evaluation
    ├── ConvolutionToGEMM.png # Illustration of how convolution operations are converted to GEMM operation before mapping onto accelerator
    ├── main.py # Selecting the accelerator and model for evaluation are set in this file. Currently, the complete logic of mapping and calculating the performance metrics happens in this file.  
    │
    ├───CNNModels #Folder contains various CNN models available for performing simulations.
    │   ├──    DenseNet121.csv
    │   ├──   GoogLeNet.csv
    │   ├──   Inception_V3.csv
    │   ├──   MobileNet_V2.csv
    │   ├──   ResNet50.csv
    │   ├──   ShuffleNet_V2.csv
    │   ├──   VGG16.csv
    │   ├──   VGG19.csv
   
```
