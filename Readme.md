# TVM Study

使用TVM优化onnx模型，用于学习演示

## 使用流程

安装 TVM ： 可参考官网的安装流程

下载模型： 

```shell
wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_onnx_0.2.0.zip
```

解压后将 hifigan_csmsc.onnx 放入到 optimize.py 同级目录

运行优化脚本

```shell
python optimize.py
```

(如果感觉优化时间比较长的话，可以修改 24行， 目前是优化 800 * num(tasks) 次， 也可以设置 eraly_stop)

测试优化后的效果

```shell
python test-tvm.py
```

测试 onnxruntime

```shell
python test-ort.py
```

## 测试结果

onnxruntime:

```text
Voc 耗时：0.33245420455932617s
Voc 耗时：0.4106733798980713s
Voc 耗时：0.56465744972229s
Voc 耗时：0.34522414207458496s
Voc 耗时：0.2857937812805176s
Voc 耗时：0.3806724548339844s
Voc 耗时：0.31762027740478516s
Voc 耗时：0.3434598445892334s
Voc 耗时：0.38057708740234375s
Voc 耗时：0.3155345916748047s
Voc 耗时：0.3306148052215576s
Voc 耗时：0.3850517272949219s
Voc 耗时：0.31652116775512695s
Voc 耗时：0.2878425121307373s
Voc 耗时：0.29862332344055176s
Voc 耗时：0.32099366188049316s
Voc 耗时：0.2700495719909668s
Voc 耗时：0.23760080337524414s
Voc 耗时：0.38071775436401367s
平均耗时:  {0.34235171267860814}
```


优化后的tvm 耗时对比

```text
Evaluate inference time cost...
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  404.7894     265.5310     849.3021     255.0272     228.1248  
               
Evaluate inference time cost...
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  262.4626     36.5496     1159.7233     32.9889      448.6531  
               
Voc 耗时：0.04015946388244629s
Voc 耗时：0.09126520156860352s
Voc 耗时：0.033791303634643555s
Voc 耗时：0.03280973434448242s
Voc 耗时：0.0339961051940918s
Voc 耗时：0.03337454795837402s
Voc 耗时：0.07142090797424316s
Voc 耗时：0.03416609764099121s
Voc 耗时：0.03380107879638672s
Voc 耗时：0.033498525619506836s
Voc 耗时：0.03378152847290039s
Voc 耗时：0.03342294692993164s
Voc 耗时：0.03389477729797363s
Voc 耗时：0.034362077713012695s
Voc 耗时：0.03422284126281738s
Voc 耗时：0.03375434875488281s
Voc 耗时：0.034120798110961914s
Voc 耗时：0.034440040588378906s
Voc 耗时：0.033980369567871094s
平均耗时:  {0.03917172080592105}
```

(启动时会有较大的波动，运行几次后稳定，所以写了单独的时间验证)


