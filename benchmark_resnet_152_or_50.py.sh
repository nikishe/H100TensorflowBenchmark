# Import tensorflow and other bits
import argparse
import tensorflow as tf
import time
import pandas as pd

#nvidia smi imports
import os
import subprocess
import sys
import json
from pathlib import Path


argParser = argparse.ArgumentParser()
argParser.add_argument("-b",  "--batch_size" , type=int, default=128, choices=[32, 64, 128, 256, 512, 4096], help="Batch size: 32, 64, 128, 256 or 512")
argParser.add_argument("-j",  "--jobID" , default="No_id",  help="The Slurm Job ID")
argParser.add_argument("-m",  "--resModel" , default="No_id",  help="Resnet 50 or resnet 152")
argParser.add_argument("-r",  "--runNumber" , default="0",  help="Resnet 50 or resnet 152")
# Parse the command-line arguments
args = argParser.parse_args()
batchSize = args.batch_size
jobID= args.jobID
resnetModel= args.resModel
runNumber= args.runNumber

def benchmark(model, params):
    """
    Benchmark Function. Enter a model and a dictionary of parameters. Current parameters used are:
    'mixed_precision': bool
    'jit_compile': bool
    'batch_size: int (recommend keeping to powers of 2, i.e. 32, 64, 128, etc. )
    'epochs': int
    """
    

    print("Begin")

    # Turn on mixed precision. Can sometimes cause training to hang.
    if params["mixed_precision"]:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed Precision Turned On")

    # Load data
    (train_images, train_labels), (
        _,
        _,
    ) = tf.keras.datasets.cifar10.load_data()
    print("Data Loaded")

    # Minimal prepocessing on data
    train_images = train_images / 255.0
    print("Data Processed")

    start_model_compile = time.time() 
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        jit_compile=params["jit_compile"],
    )
    finish_model_compile =  time.time() - start_model_compile 
    print("The compilation of the model took:" + str(finish_model_compile))

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
      except RuntimeError as e:
        print(e)

    start = time.time()
    
    # Training
    print("Training Begin")
    model.fit(
        x=train_images,
        y=train_labels,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
    )

    # Record time taken for full script and add to a dataframe along with parameters
    print("Training Finished")
    time_taken_s = time.time() - start
    results = pd.DataFrame(params, index=[0])
    results["time_taken_s"] = time_taken_s
    print("Finished")
    return results


if __name__ == "__main__":
    # Load resnet model
    start_model_create = time.time()
    if resnetModel=="152":
        os.system("echo Model =====ResNet152v2")
        print("Model =====ResNet152V2")
        model = tf.keras.applications.resnet_v2.ResNet152V2(
            include_top=True,
            weights=None,
            input_tensor=None,
            #input_shape=(32, 32, 3),
            input_shape=(256, 256, 3),
            pooling=None,
            classes=10,
        )
    elif resnetModel=="50":
        os.system("echo Model =====ResNet50")
        print("Model =====ResNet50")
        model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling=None,
        classes=10,
        )
    else:
        print(type(resnetModel))
        os.system("echo No nodel found")
    finish_model_create =  time.time() - start_model_create 
    print("The creation of the model took:" + str(finish_model_create))
    #opts, args = args.getopt(argv,"hi:o:",["bs=","j="]) 
    # Test 1
    params = {
        "batch_size": batchSize,
        "jit_compile": False,
        "epochs": 20,
        "mixed_precision": False,
    }
    df1 = benchmark(model, params)
    
        # Test 2
    params = {
        "batch_size": batchSize,
        "jit_compile": True,
        "epochs": 20,
        "mixed_precision": False,
    }
    df2 = benchmark(model, params)
    
    
    # Collect results in one dataframe and save to csv for later plotting
    results = pd.concat([df1,df2])
    results.to_csv("Benchmark_Results_resnet152_"+str(batchSize)+"_"+str(jobID)+"_Run_"+str(runNumber)+".csv")
    lines = subprocess.check_output(['nvidia-smi', '-q']).decode(sys.stdout.encoding).split(os.linesep)
    path = Path('Run_'+str(runNumber)+'_NvidiaSMI_Stats.txt')
    path.write_text('\n'.join(lines))

