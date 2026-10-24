# Circle graph observations

All models loaded with `CircleDocument.from_bytes` and passed `document.verify()` unless issues listed. Cache Concat identification requires axis 1 and exact decode edge shapes; K additionally feeds transpose, V directly feeds attention matmul. Operator order is not used for identification.

| artifact | operators | all QUANTIZE | Concat edge QUANTIZE | delta edge QUANTIZE | all KV edge QUANTIZE |
|---|---:|---:|---:|---:|---:|
| decoder_decode_forced_noopt.circle | 60 | 4 | 0 | 2 | 2 |
| decoder_decode_forced_o1.circle | 58 | 4 | 0 | 2 | 2 |
| decoder_decode_natural_noopt.circle | 58 | 2 | 0 | 0 | 0 |
| decoder_decode_natural_o1.circle | 56 | 2 | 0 | 0 | 0 |
| decoder_prefill_forced_noopt.circle | 58 | 6 | 0 | 2 | 4 |
| decoder_prefill_forced_o1.circle | 56 | 6 | 0 | 2 | 4 |
| decoder_prefill_natural_noopt.circle | 54 | 2 | 0 | 0 | 0 |
| decoder_prefill_natural_o1.circle | 52 | 2 | 0 | 0 | 0 |

## decoder_decode_forced_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 50:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 77 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 59:ADD | GRAPH_OUTPUT |
| delta K | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |
| delta V | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| delta K branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |
| delta K branch | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | 45:QUANTIZE |
| delta K graph output | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| delta V branch | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:CONCATENATION | 47:RESHAPE |
| delta V branch | 65 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 47:RESHAPE | 48:QUANTIZE |
| delta V graph output | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit


Operator 43:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |

Operator 46:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 46:CONCATENATION |
| output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:CONCATENATION | 47:RESHAPE |

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 50:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 45; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | 45:QUANTIZE |
| QUANTIZE output | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 45:QUANTIZE | GRAPH_OUTPUT |

Operator 48; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 65 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 47:RESHAPE | 48:QUANTIZE |
| QUANTIZE output | 66 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 48:QUANTIZE | GRAPH_OUTPUT |

Operator 51; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 68 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 50:ADD | 51:QUANTIZE, 59:ADD |
| QUANTIZE output | 69 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 51:QUANTIZE | 52:RMS_NORM |

## decoder_decode_forced_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 75 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |
| delta V | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.3125] | [91] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.3125] | [91] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| delta K branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K branch | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| delta K graph output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| delta V branch | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |
| delta V branch | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| delta V graph output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit


Operator 41:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.3125] | [91] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |

Operator 44:CONCATENATION has input/output affine grids that differ; structural verification does not check this semantic invariant.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.3125] | [91] | 5:FULLY_CONNECTED | 14:CONCATENATION, 44:CONCATENATION |
| output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 43; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| QUANTIZE output | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_49 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

Operator 46; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| QUANTIZE output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_50 | [1, 1, 1, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 66 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 67 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_decode_natural_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 75 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | GRAPH_OUTPUT |
| delta V | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:RESHAPE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.004473801702260971] | [128] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 45:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.012130454182624817] | [104] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 43:CONCATENATION |
| delta K branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 43:CONCATENATION | 44:RESHAPE |
| delta K graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 44:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 45:CONCATENATION |
| delta V branch | 63 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 45:CONCATENATION | 46:RESHAPE |
| delta V graph output | 64 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 46:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 66 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 67 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_decode_natural_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| graph input | 10 | tico::attention_mask | [1, 1, 8] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 24:ADD, 36:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 1, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 19:MUL, 31:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 1, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 20:MUL, 32:MUL |
| graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 73 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 1, 16] | UINT8 | [0.024770094081759453] | [112] | 55:ADD | GRAPH_OUTPUT |
| delta K | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | GRAPH_OUTPUT |
| delta V | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:RESHAPE | GRAPH_OUTPUT |

### K Concat operator 13

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['22:TRANSPOSE', '34:TRANSPOSE']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| K past cache graph input | 13 | tico::past_key_value_0 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | GRAPH_INPUT | 11:RESHAPE |
| K past concat input | 27 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_21 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 11:RESHAPE | 13:CONCATENATION |
| K current concat input | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| K concat output | 29 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_1 | [1, 8, 8] | UINT8 | [0.004473801702260971] | [128] | 13:CONCATENATION | 22:TRANSPOSE, 34:TRANSPOSE |

### V Concat operator 14

axis=1; [[1, 7, 8], [1, 1, 8]] → [1, 8, 8]; downstream=['26:BATCH_MATMUL', '38:BATCH_MATMUL']; equal input/output qparams=True; cache-edge Quantize ops=[].

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| V past cache graph input | 14 | tico::past_key_value_1 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | GRAPH_INPUT | 12:RESHAPE |
| V past concat input | 28 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_22 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 12:RESHAPE | 14:CONCATENATION |
| V current concat input | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 43:CONCATENATION |
| V concat output | 30 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_2 | [1, 8, 8] | UINT8 | [0.012130454182624817] | [104] | 14:CONCATENATION | 26:BATCH_MATMUL, 38:BATCH_MATMUL |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 13:CONCATENATION, 41:CONCATENATION |
| delta K branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_6 | [1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K graph output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 1, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 20 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 14:CONCATENATION, 43:CONCATENATION |
| delta V branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_7 | [1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 43:CONCATENATION | 44:RESHAPE |
| delta V graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 1, 8] | UINT8 | [0.012130454182624817] | [104] | 44:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 1, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| QUANTIZE output | 15 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 1, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 47; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 64 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 1, 16] | FLOAT32 | [] | [] | 46:ADD | 47:QUANTIZE, 55:ADD |
| QUANTIZE output | 65 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_51 | [1, 1, 16] | UINT8 | [0.02521858550608158] | [115] | 47:QUANTIZE | 48:RMS_NORM |

## decoder_prefill_forced_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 22:ADD, 34:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 17:MUL, 29:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 18:MUL, 30:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 73 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 57:ADD | GRAPH_OUTPUT |
| delta K | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |
| delta V | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 41:CONCATENATION |
| delta K branch | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 41:CONCATENATION | 42:RESHAPE |
| delta K branch | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| delta K graph output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 44:CONCATENATION |
| delta V branch | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 44:CONCATENATION | 45:RESHAPE |
| delta V branch | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| delta V graph output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 48:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 11; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 41:CONCATENATION |
| QUANTIZE output | 25 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:QUANTIZE | 20:TRANSPOSE, 32:TRANSPOSE |

Operator 12; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 44:CONCATENATION |
| QUANTIZE output | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_23 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:QUANTIZE | 24:BATCH_MATMUL, 36:BATCH_MATMUL |

Operator 43; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 42:RESHAPE | 43:QUANTIZE |
| QUANTIZE output | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 43:QUANTIZE | GRAPH_OUTPUT |

Operator 46; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 61 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 45:RESHAPE | 46:QUANTIZE |
| QUANTIZE output | 62 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 46:QUANTIZE | GRAPH_OUTPUT |

Operator 49; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 64 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 48:ADD | 49:QUANTIZE, 57:ADD |
| QUANTIZE output | 65 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 49:QUANTIZE | 50:RMS_NORM |

## decoder_prefill_forced_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 22:ADD, 34:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 17:MUL, 29:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 18:MUL, 30:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 71 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 55:ADD | GRAPH_OUTPUT |
| delta K | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |
| delta V | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 39:CONCATENATION |
| delta K branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 39:CONCATENATION | 40:RESHAPE |
| delta K branch | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | 41:QUANTIZE |
| delta K graph output | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 42:CONCATENATION |
| delta V branch | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:CONCATENATION | 43:RESHAPE |
| delta V branch | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 43:RESHAPE | 44:QUANTIZE |
| delta V graph output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 46:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 11; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 11:QUANTIZE, 39:CONCATENATION |
| QUANTIZE output | 25 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_22 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 11:QUANTIZE | 20:TRANSPOSE, 32:TRANSPOSE |

Operator 12; on Concat edge=False; on delta output edge=False; on prefill present compute edge=True.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 12:QUANTIZE, 42:CONCATENATION |
| QUANTIZE output | 26 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_23 | [1, 7, 8] | UINT8 | [0.3125] | [91] | 12:QUANTIZE | 24:BATCH_MATMUL, 36:BATCH_MATMUL |

Operator 41; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | 41:QUANTIZE |
| QUANTIZE output | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_47 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 41:QUANTIZE | GRAPH_OUTPUT |

Operator 44; on Concat edge=False; on delta output edge=True; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 59 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 43:RESHAPE | 44:QUANTIZE |
| QUANTIZE output | 60 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::quantize_per_tensor_48 | [1, 1, 7, 8] | UINT8 | [0.0625] | [113] | 44:QUANTIZE | GRAPH_OUTPUT |

Operator 47; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 62 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 46:ADD | 47:QUANTIZE, 55:ADD |
| QUANTIZE output | 63 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 47:QUANTIZE | 48:RMS_NORM |

## decoder_prefill_natural_noopt.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 44:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 20:ADD, 32:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 15:MUL, 27:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 16:MUL, 28:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 69 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 53:ADD | GRAPH_OUTPUT |
| delta K | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | GRAPH_OUTPUT |
| delta V | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:RESHAPE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 18:TRANSPOSE, 30:TRANSPOSE, 39:CONCATENATION |
| delta K branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 39:CONCATENATION | 40:RESHAPE |
| delta K graph output | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 40:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 22:BATCH_MATMUL, 34:BATCH_MATMUL, 41:CONCATENATION |
| delta V branch | 57 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 41:CONCATENATION | 42:RESHAPE |
| delta V graph output | 58 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 42:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 44:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 45; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 60 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 44:ADD | 45:QUANTIZE, 53:ADD |
| QUANTIZE output | 61 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 45:QUANTIZE | 46:RMS_NORM |

## decoder_prefill_natural_o1.circle

Verification: no issues.

Graph inputs:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| graph input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 42:ADD |
| graph input | 10 | tico::attention_mask | [1, 7, 7] | UINT8 | [0.47058823704719543] | [255] | GRAPH_INPUT | 20:ADD, 32:ADD |
| graph input | 11 | tico::position_embeddings_0 | [1, 7, 8] | UINT8 | [0.0078038922511041164] | [127] | GRAPH_INPUT | 8:MUL, 15:MUL, 27:MUL |
| graph input | 12 | tico::position_embeddings_1 | [1, 7, 8] | UINT8 | [0.007520975079387426] | [127] | GRAPH_INPUT | 9:MUL, 16:MUL, 28:MUL |

Graph outputs (delta cache):

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| hidden | 67 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_6 | [1, 7, 16] | UINT8 | [0.024770094081759453] | [112] | 51:ADD | GRAPH_OUTPUT |
| delta K | 54 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 38:RESHAPE | GRAPH_OUTPUT |
| delta V | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 40:RESHAPE | GRAPH_OUTPUT |

### Delta output branches


K delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta K branch | 24 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::add | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 10:ADD | 18:TRANSPOSE, 30:TRANSPOSE, 37:CONCATENATION |
| delta K branch | 53 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_4 | [1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 37:CONCATENATION | 38:RESHAPE |
| delta K graph output | 54 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_17 | [1, 1, 7, 8] | UINT8 | [0.004473801702260971] | [128] | 38:RESHAPE | GRAPH_OUTPUT |

V delta edge from producer to graph output:

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| delta V branch | 18 | tico.quantization.wrapq.wrappers.nn.quant_linear.QuantLinear::linear_2 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 5:FULLY_CONNECTED | 22:BATCH_MATMUL, 34:BATCH_MATMUL, 39:CONCATENATION |
| delta V branch | 55 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::cat_5 | [1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 39:CONCATENATION | 40:RESHAPE |
| delta V graph output | 56 | tico.quantization.wrapq.wrappers.llama.quant_attention.QuantLlamaAttention::reshape_default_18 | [1, 1, 7, 8] | UINT8 | [0.012130454182624817] | [104] | 40:RESHAPE | GRAPH_OUTPUT |

### Value-preserving qparam audit

No qparam mismatch on audited Concat/view data edges.

### Every QUANTIZE


Operator 0; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 9 | tico::hidden_states | [1, 7, 16] | FLOAT32 | [] | [] | GRAPH_INPUT | 0:QUANTIZE, 42:ADD |
| QUANTIZE output | 13 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor | [1, 7, 16] | UINT8 | [0.024814432486891747] | [116] | 0:QUANTIZE | 1:RMS_NORM |

Operator 43; on Concat edge=False; on delta output edge=False; on prefill present compute edge=False.

| role | tensor index | name | shape | dtype | scale | zero point | producer | consumers |
|---|---:|---|---|---|---|---|---|---|
| QUANTIZE input | 58 | tico.quantization.wrapq.wrappers.llama.quant_decoder_layer.QuantLlamaDecoderLayer::add_5 | [1, 7, 16] | FLOAT32 | [] | [] | 42:ADD | 43:QUANTIZE, 51:ADD |
| QUANTIZE output | 59 | tico.quantization.wrapq.wrappers.ops.quant_rmsnorm.QuantRMSNorm::quantize_per_tensor_49 | [1, 7, 16] | UINT8 | [0.02521858550608158] | [115] | 43:QUANTIZE | 44:RMS_NORM |

# Cache ABI comparison

| scenario / optimization | prefill K → decode past K | prefill V → decode past V | decode delta K → next past K | decode delta V → next past V |
|---|---|---|---|---|
| natural / noopt | identical (class 1) | identical (class 1) | identical (class 1) | identical (class 1) |
| natural / o1 | identical (class 1) | identical (class 1) | identical (class 1) | identical (class 1) |
| forced / noopt | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} |
| forced / o1 | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} | different: {'dtype': 'UINT8', 'scale': [0.0625], 'zero_point': [113]} → {'dtype': 'UINT8', 'scale': [0.3125], 'zero_point': [91]} |

# O1 comparison

Every O1 model has exactly two fewer operators. Comparing producer output tensor names identifies the removed operators as the attention-head layout round trip: `reshape_default_15` (shape [1,Q,2,8]) and `reshape_default_16` (shape [1,Q,16]). No QUANTIZE is removed. `SPLIT_V` is canonicalized to `SPLIT` (three instances). Cache input ABI and cache grids do not change. Forced decode retains the delta-branch qparam mismatch in both noopt and O1.

prefill/natural: removed noopt operators 36:RESHAPE (['reshape_default_15']), 37:RESHAPE (['reshape_default_16']).
decode/natural: removed noopt operators 40:RESHAPE (['reshape_default_15']), 41:RESHAPE (['reshape_default_16']).
prefill/forced: removed noopt operators 38:RESHAPE (['reshape_default_15']), 39:RESHAPE (['reshape_default_16']).
decode/forced: removed noopt operators 40:RESHAPE (['reshape_default_15']), 41:RESHAPE (['reshape_default_16']).

# Observed topology and ABI interpretation

The primary past/current decode concatenations are locally well formed: K and V each have identical input/output quantization parameters. Both noopt and O1 implement Case B on the past edge and Case C on the current edge. There is no Circle QUANTIZE on either edge.

```text
noopt, forced decode (main compute path)
  past graph input UINT8 C (.3125,91) -> RESHAPE C --+
                                                   +-> CONCAT axis1 C -> attention
  K ADD / V FULLY_CONNECTED output C --------------+

o1, forced decode (main compute path)
  past graph input UINT8 C (.3125,91) -> RESHAPE C --+
                                                   +-> CONCAT axis1 C -> attention
  K ADD / V FULLY_CONNECTED output C --------------+
```

The controlled model also exposes a separate Case D issue on its delta-output side branch. The shared K ADD / V FULLY_CONNECTED is retargeted to C on the compute edge, while the single-input Concat that implements the stack for the delta branch retains its earlier natural producer grid P. A value-preserving single-input Concat cannot implement the represented P transition. The following reshape retains P and an explicit Quantize then produces N. O1 preserves this mismatch.

```text
forced decode delta branch (both noopt/O1)
  current producer C -> singleton CONCAT output P -> RESHAPE P -> QUANTIZE N -> delta output
                       ^ qparam mismatch

forced prefill (both noopt/O1)
  current producer P --+-> QUANTIZE C -> attention compute
                       +-> singleton CONCAT P -> RESHAPE P -> QUANTIZE N -> delta output
```

Forced prefill output and forced decode delta output are N (.0625,113), whereas decode past inputs demand C (.3125,91). Because decode has no cache-input Quantize, this is ABI class 3: the host must explicitly requantize cache bytes N→C before passing them to decode. Copying UINT8 bytes directly interprets different real values. In this controlled same-scale-for-K/V example the host mapping is `q_C = clamp(round((q_N-113)*0.0625/0.3125)+91, 0, 255)`. Requantization reconciles storage grids but does not repair the separate stale-P metadata on the delta branch.

Natural prefill/delta outputs and natural decode past inputs have identical serialized qparams, so that measured boundary is class 1. This natural equality does not establish that arbitrary independently calibrated or deliberately distinct observer domains are chainable.

The reference fake-quantize runtime can pass values through the singleton Concat despite inconsistent P metadata because value-preserving operations do not round to their declared output grid. Successful structural verification and fake-quantize execution therefore do not validate this quantized storage-semantics mismatch or establish integer backend correctness.
