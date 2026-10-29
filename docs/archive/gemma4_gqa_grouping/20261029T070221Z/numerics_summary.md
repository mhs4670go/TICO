All table values below are worst-case across six cases (decode/append/B2 × core/full). PEIR is percent. Default is UINT8 MinMax; explicit MinMax is repeated as a control. FP32; eval/dropout=0; one CPU thread. Actual per-case mean/max/relative L2 and PEIR are retained in JSON/CSV.

|Implementation|Experiment/policy|Output max abs vs BASE|Output PEIR % vs BASE|Weights max abs vs BASE|
|---|---|---:|---:|---:|
|base|NO_QUANT|0|0|0|
|base|fixed_vs_base/default|0|0|0|
|base|fixed_vs_base/minmax|0|0|0|
|base|fixed_vs_base/ema|0|0|0|
|base|independent_vs_base/default|0|0|0|
|base|independent_vs_base/minmax|0|0|0|
|base|independent_vs_base/ema|0|0|0|
|a|NO_QUANT|0|0|0|
|a|fixed_vs_base/default|0|0|0|
|a|fixed_vs_base/minmax|0|0|0|
|a|fixed_vs_base/ema|0|0|0|
|a|independent_vs_base/default|0|0|0|
|a|independent_vs_base/minmax|0|0|0|
|a|independent_vs_base/ema|0|0|0|
|b|NO_QUANT|3.5762787e-07|1.5300704e-05|4.1723251e-07|
|b|fixed_vs_base/default|0|0|0|
|b|fixed_vs_base/minmax|0|0|0|
|b|fixed_vs_base/ema|0|0|0|
|b|independent_vs_base/default|1.1920929e-07|6.0817455e-06|0|
|b|independent_vs_base/minmax|1.1920929e-07|6.0817455e-06|0|
|b|independent_vs_base/ema|0.37319648|35.353531|0.40353376|

|Implementation|Policy|Fixed output max abs vs FP|Fixed PEIR % vs FP|Independent output max abs vs FP|Independent PEIR % vs FP|
|---|---|---:|---:|---:|---:|
|base|default|0.19506332|8.0308215|0.19506332|8.0308215|
|base|minmax|0.19506332|8.0308215|0.19506332|8.0308215|
|base|ema|0.639801|31.849582|0.639801|31.849582|
|a|default|0.19506332|8.0308215|0.19506332|8.0308215|
|a|minmax|0.19506332|8.0308215|0.19506332|8.0308215|
|a|ema|0.639801|31.849582|0.639801|31.849582|
|b|default|0.19506332|8.0308215|0.19506326|8.0308204|
|b|minmax|0.19506332|8.0308215|0.19506326|8.0308204|
|b|ema|0.639801|31.849582|0.38277853|18.452894|
