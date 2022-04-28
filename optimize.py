# https://zhuanlan.zhihu.com/p/431462179

import logging
import sys
import time
import numpy as np
import tvm
from tvm import te
import tvm.testing
import onnx
import tvm.relay as relay
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime

# the module is called `autotvm`
from tvm import autotvm


def run_tuning(tasks, task_weights, log_file):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=800 * len(tasks),  # 800 * len(tasks)
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)


def define_task(mod, params, target):
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    return tasks, task_weights


def out_build(log_file, mod, target, params, out_so):
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
    lib.export_library(out_so)


def tune_onnx_mode(model_path, log_file, shape_dict, target, out_so):
    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    tasks, task_weights = define_task(mod, params, target)
    # 开始finetune
    run_tuning(tasks, task_weights, log_file)
    # 导出log_file
    out_build(log_file, mod, target, params, out_so)


if __name__ == '__main__':

    toune_tasks = []

    # melgan
    for size in [64]:
        tasks = {
            "model_path": f"hifigan_csmsc.onnx",
            "log_file": f"hifigan_csmsc_{size}.log",
            "shape_dict": {
                "logmel": (size, 80)
                           },
            "target": "llvm -mcpu=core-avx2",
            "out_so": f"hifigan_csmsc_{size}.so",
        }
        toune_tasks.append(tasks)

    # start:
    for task in toune_tasks:
        
        tune_onnx_mode(model_path=task['model_path'],
                           log_file=task['log_file'],
                           shape_dict=task['shape_dict'],
                           target=task['target'],
                           out_so=task['out_so'])


