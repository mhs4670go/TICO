|Fixture|Impl|Mode|Quant|EP QK/PV|Circle QK/PV|Concat|Reshape|Transpose|Slice|Softmax|Q/DQ|Float activations|Runtime|max abs output0|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|custom|base|prefill|False|24/24|24/24|3|47|27|40|24|0/0|234|FAIL|FAIL|
|custom|base|decode|False|24/24|24/24|5|50|24|40|24|0/0|238|FAIL|FAIL|
|custom|base|prefill|True|24/24|24/24|3|48|27|40|24|10/0|0|EXECUTED|0.0|
|custom|base|decode|True|24/24|24/24|5|51|24|40|24|12/0|0|EXECUTED|0.0|
|tiny|base|prefill|False|4/4|4/4|3|15|7|8|4|0/0|70|FAIL|FAIL|
|tiny|base|decode|False|4/4|4/4|5|18|4|8|4|0/0|74|FAIL|FAIL|
|tiny|base|prefill|True|4/4|4/4|3|16|7|8|4|10/0|0|EXECUTED|0.0|
|tiny|base|decode|True|4/4|4/4|5|19|4|8|4|12/0|0|EXECUTED|0.0|
|custom|a|prefill|False|24/8|24/8|11|48|27|64|24|0/0|251|EXECUTED|5.960464477539062e-07|
|custom|a|decode|False|24/8|24/8|13|51|24|64|24|0/0|255|EXECUTED|1.9371509552001953e-07|
|custom|a|prefill|True|24/8|24/8|11|48|27|64|24|10/0|0|EXECUTED|0.0|
|custom|a|decode|True|24/8|24/8|13|51|24|64|24|12/0|0|EXECUTED|0.0|
|tiny|a|prefill|False|4/2|4/2|5|16|7|12|4|0/0|75|EXECUTED|2.384185791015625e-07|
|tiny|a|decode|False|4/2|4/2|7|19|4|12|4|0/0|79|EXECUTED|1.1548399925231934e-07|
|tiny|a|prefill|True|4/2|4/2|5|16|7|12|4|10/0|0|EXECUTED|0.0|
|tiny|a|decode|True|4/2|4/2|7|19|4|12|4|12/0|0|EXECUTED|0.0|
|custom|b|prefill|False|8/8|8/8|12|80|11|64|8|0/0|220|EXECUTED|5.960464477539062e-07|
|custom|b|decode|False|8/8|8/8|13|82|8|64|8|0/0|222|FAIL|FAIL|
|custom|b|prefill|True|8/8|8/8|12|80|11|64|8|10/0|0|EXECUTED|0.0|
|custom|b|decode|True|8/8|8/8|13|83|8|64|8|12/0|0|EXECUTED|0.0|
|tiny|b|prefill|False|2/2|2/2|6|22|5|12|2|0/0|74|EXECUTED|2.384185791015625e-07|
|tiny|b|decode|False|2/2|2/2|7|24|2|12|2|0/0|76|FAIL|FAIL|
|tiny|b|prefill|True|2/2|2/2|6|22|5|12|2|10/0|0|EXECUTED|0.0|
|tiny|b|decode|True|2/2|2/2|7|25|2|12|2|12/0|0|EXECUTED|0.0|

Default O1 retained. FP BASE prefill/decode and B decode Circle reference runtime fail serialized rank contract: Softmax is rank4 but PV BMM output contract rank3. Eager/torch.export are valid.

Quantized runtime uses explicit FAKE_QUANTIZE semantic reference, not bit-exact NPU execution. ABI inputs are aligned to serialized grids before eager comparison.

The initial per-head PV-output marker left grouped MatMul outputs FLOAT32; final wrapper-local scalar-affine group marker removes all such float islands without extra Q/DQ.
