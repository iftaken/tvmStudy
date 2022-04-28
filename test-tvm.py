import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
import onnx
import numpy as np
import time

dtype = "float32"
input_name = "logmel"
shape_dict = {
            input_name: (64, 80)
            }
data_shape = shape_dict['logmel']
model_path = "hifigan_csmsc.onnx"

log_file = "hifigan_csmsc_64.log"


def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=30, repeat=5))

def compute_before(model_path, shape_dict):
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    
    target = "llvm -mcpu=core-avx2"
    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

def compute_log(model_path, shape_dict):
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    target = "llvm -mcpu=core-avx2"
    Input, result = tvm.auto_scheduler.load_best_record(log_file)
    print(Input)
    print(result)
        # print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    evaluate_performance(lib, data_shape)

    # compile kernels with graph-level best records
    # print("\nEvaluation of the network been tuned on graph level:")
    # with autotvm.apply_graph_best(graph_opt_sch_file):
    #     print("Compile...")
    #     with tvm.transform.PassContext(opt_level=3):
    #         lib = relay.build_module.build(mod, target=target, params=params)
    #     evaluate_performance(lib, data_shape)

def compute_so(so_path):
    target = "llvm -mcpu=core-avx2"
    # dev = tvm.device(str(target), 0)
    lib = tvm.runtime.load_module(so_path)
    evaluate_performance(lib, data_shape)
    
def compute_so_local(so_path):
    lib = tvm.runtime.load_module(so_path)
    # evaluate_performance(lib, data_shape)
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)
    
    diffs = []
   
    for i in range(30):
        t1 = time.time()
        module.run()
        t2 = time.time()
        if i > 10:
            diffs.append(t2 - t1)
            print(f"Voc 耗时：{t2 - t1}s")
    print("平均耗时: ", {sum(diffs) / len(diffs)})    


compute_before(model_path, shape_dict)
compute_so("hifigan_csmsc_64.so")
compute_so_local("hifigan_csmsc_64.so")