import paddlespeech
import onnxruntime as ort
import onnx
import numpy as np
import time


def get_sess(model_path, sess_conf: dict=None):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if "gpu" in sess_conf["device"]:
        # fastspeech2/mb_melgan can't use trt now!
        if sess_conf["use_trt"]:
            providers = ['TensorrtExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']
    elif sess_conf["device"] == "cpu":
        providers = ['CPUExecutionProvider']
    sess_options.intra_op_num_threads = sess_conf["cpu_threads"]
    sess = ort.InferenceSession(
        model_path, providers=providers, sess_options=sess_options)
    return sess


def inference(model_path):
    conf = {
        "device": "cpu",
        "cpu_threads": 80
    }
    sess_options = get_sess(model_path, conf)
    mel_chunk = np.random.rand(64, 80)
    mel_chunk = mel_chunk.astype(np.float32)
    diffs = []
    for i in range(30):
        t1 = time.time()
        sub_wav = sess_options.run(
                            output_names=None, input_feed={'logmel': mel_chunk})
        t2 = time.time()
        if i > 10:
            diffs.append(t2 - t1)
            print(f"Voc 耗时：{t2 - t1}s")
    print("平均耗时: ", {sum(diffs) / len(diffs)})
    print(sub_wav)
    return sub_wav

model_path = "hifigan_csmsc.onnx"
inference(model_path)





