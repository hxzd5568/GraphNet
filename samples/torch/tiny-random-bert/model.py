
class GraphModule(torch.nn.Module):

    def forward(self, p_bert_embeddings_word_embeddings_weight, p_bert_embeddings_token_type_embeddings_weight, p_bert_embeddings_position_embeddings_weight, p_bert_embeddings_layernorm_weight, p_bert_embeddings_layernorm_bias, p_bert_encoder_layer_0_attention_self_query_weight, p_bert_encoder_layer_0_attention_self_query_bias, p_bert_encoder_layer_0_attention_self_key_weight, p_bert_encoder_layer_0_attention_self_key_bias, p_bert_encoder_layer_0_attention_self_value_weight, p_bert_encoder_layer_0_attention_self_value_bias, p_bert_encoder_layer_0_attention_output_dense_weight, p_bert_encoder_layer_0_attention_output_dense_bias, p_bert_encoder_layer_0_attention_output_layernorm_weight, p_bert_encoder_layer_0_attention_output_layernorm_bias, p_bert_encoder_layer_0_intermediate_dense_weight, p_bert_encoder_layer_0_intermediate_dense_bias, p_bert_encoder_layer_0_output_dense_weight, p_bert_encoder_layer_0_output_dense_bias, p_bert_encoder_layer_0_output_layernorm_weight, p_bert_encoder_layer_0_output_layernorm_bias, p_bert_encoder_layer_1_attention_self_query_weight, p_bert_encoder_layer_1_attention_self_query_bias, p_bert_encoder_layer_1_attention_self_key_weight, p_bert_encoder_layer_1_attention_self_key_bias, p_bert_encoder_layer_1_attention_self_value_weight, p_bert_encoder_layer_1_attention_self_value_bias, p_bert_encoder_layer_1_attention_output_dense_weight, p_bert_encoder_layer_1_attention_output_dense_bias, p_bert_encoder_layer_1_attention_output_layernorm_weight, p_bert_encoder_layer_1_attention_output_layernorm_bias, p_bert_encoder_layer_1_intermediate_dense_weight, p_bert_encoder_layer_1_intermediate_dense_bias, p_bert_encoder_layer_1_output_dense_weight, p_bert_encoder_layer_1_output_dense_bias, p_bert_encoder_layer_1_output_layernorm_weight, p_bert_encoder_layer_1_output_layernorm_bias, p_bert_encoder_layer_2_attention_self_query_weight, p_bert_encoder_layer_2_attention_self_query_bias, p_bert_encoder_layer_2_attention_self_key_weight, p_bert_encoder_layer_2_attention_self_key_bias, p_bert_encoder_layer_2_attention_self_value_weight, p_bert_encoder_layer_2_attention_self_value_bias, p_bert_encoder_layer_2_attention_output_dense_weight, p_bert_encoder_layer_2_attention_output_dense_bias, p_bert_encoder_layer_2_attention_output_layernorm_weight, p_bert_encoder_layer_2_attention_output_layernorm_bias, p_bert_encoder_layer_2_intermediate_dense_weight, p_bert_encoder_layer_2_intermediate_dense_bias, p_bert_encoder_layer_2_output_dense_weight, p_bert_encoder_layer_2_output_dense_bias, p_bert_encoder_layer_2_output_layernorm_weight, p_bert_encoder_layer_2_output_layernorm_bias, p_bert_encoder_layer_3_attention_self_query_weight, p_bert_encoder_layer_3_attention_self_query_bias, p_bert_encoder_layer_3_attention_self_key_weight, p_bert_encoder_layer_3_attention_self_key_bias, p_bert_encoder_layer_3_attention_self_value_weight, p_bert_encoder_layer_3_attention_self_value_bias, p_bert_encoder_layer_3_attention_output_dense_weight, p_bert_encoder_layer_3_attention_output_dense_bias, p_bert_encoder_layer_3_attention_output_layernorm_weight, p_bert_encoder_layer_3_attention_output_layernorm_bias, p_bert_encoder_layer_3_intermediate_dense_weight, p_bert_encoder_layer_3_intermediate_dense_bias, p_bert_encoder_layer_3_output_dense_weight, p_bert_encoder_layer_3_output_dense_bias, p_bert_encoder_layer_3_output_layernorm_weight, p_bert_encoder_layer_3_output_layernorm_bias, p_bert_encoder_layer_4_attention_self_query_weight, p_bert_encoder_layer_4_attention_self_query_bias, p_bert_encoder_layer_4_attention_self_key_weight, p_bert_encoder_layer_4_attention_self_key_bias, p_bert_encoder_layer_4_attention_self_value_weight, p_bert_encoder_layer_4_attention_self_value_bias, p_bert_encoder_layer_4_attention_output_dense_weight, p_bert_encoder_layer_4_attention_output_dense_bias, p_bert_encoder_layer_4_attention_output_layernorm_weight, p_bert_encoder_layer_4_attention_output_layernorm_bias, p_bert_encoder_layer_4_intermediate_dense_weight, p_bert_encoder_layer_4_intermediate_dense_bias, p_bert_encoder_layer_4_output_dense_weight, p_bert_encoder_layer_4_output_dense_bias, p_bert_encoder_layer_4_output_layernorm_weight, p_bert_encoder_layer_4_output_layernorm_bias, p_bert_pooler_dense_weight, p_bert_pooler_dense_bias, p_classifier_weight, p_classifier_bias, b_bert_embeddings_position_ids, input_ids, token_type_ids, attention_mask):
        slice_1 = torch.ops.aten.slice.Tensor(b_bert_embeddings_position_ids, 0, 0, 9223372036854775807);  b_bert_embeddings_position_ids = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 12);  slice_1 = None
        embedding = torch.ops.aten.embedding.default(p_bert_embeddings_word_embeddings_weight, input_ids, 0);  p_bert_embeddings_word_embeddings_weight = input_ids = None
        embedding_1 = torch.ops.aten.embedding.default(p_bert_embeddings_token_type_embeddings_weight, token_type_ids);  p_bert_embeddings_token_type_embeddings_weight = token_type_ids = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(p_bert_embeddings_position_embeddings_weight, slice_2);  p_bert_embeddings_position_embeddings_weight = slice_2 = None
        add_ = torch.ops.aten.add_.Tensor(add, embedding_2);  add = embedding_2 = None
        layer_norm = torch.ops.aten.layer_norm.default(add_, [32], p_bert_embeddings_layernorm_weight, p_bert_embeddings_layernorm_bias, 1e-12);  add_ = p_bert_embeddings_layernorm_weight = p_bert_embeddings_layernorm_bias = None
        dropout = torch.ops.aten.dropout.default(layer_norm, 0.1, False);  layer_norm = None
        slice_3 = torch.ops.aten.slice.Tensor(attention_mask, 0, 0, 9223372036854775807);  attention_mask = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        slice_4 = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
        expand = torch.ops.aten.expand.default(slice_4, [1, 1, 12, 12]);  slice_4 = None
        to = torch.ops.aten.to.dtype(expand, torch.float32);  expand = None
        rsub = torch.ops.aten.rsub.Scalar(to, 1.0);  to = None
        to_1 = torch.ops.aten.to.dtype(rsub, torch.bool)
        masked_fill = torch.ops.aten.masked_fill.Scalar(rsub, to_1, -3.4028234663852886e+38);  rsub = to_1 = None
        linear = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_query_weight, p_bert_encoder_layer_0_attention_self_query_bias);  p_bert_encoder_layer_0_attention_self_query_weight = p_bert_encoder_layer_0_attention_self_query_bias = None
        view = torch.ops.aten.view.default(linear, [1, 12, 4, 8]);  linear = None
        permute = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        linear_1 = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_key_weight, p_bert_encoder_layer_0_attention_self_key_bias);  p_bert_encoder_layer_0_attention_self_key_weight = p_bert_encoder_layer_0_attention_self_key_bias = None
        view_1 = torch.ops.aten.view.default(linear_1, [1, 12, 4, 8]);  linear_1 = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        linear_2 = torch.ops.aten.linear.default(dropout, p_bert_encoder_layer_0_attention_self_value_weight, p_bert_encoder_layer_0_attention_self_value_bias);  p_bert_encoder_layer_0_attention_self_value_weight = p_bert_encoder_layer_0_attention_self_value_bias = None
        view_2 = torch.ops.aten.view.default(linear_2, [1, 12, 4, 8]);  linear_2 = None
        permute_2 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        scaled_dot_product_attention = torch.ops.aten.scaled_dot_product_attention.default(permute, permute_1, permute_2, masked_fill);  permute = permute_1 = permute_2 = None
        transpose = torch.ops.aten.transpose.int(scaled_dot_product_attention, 1, 2);  scaled_dot_product_attention = None
        reshape = torch.ops.aten.reshape.default(transpose, [1, 12, 32]);  transpose = None
        linear_3 = torch.ops.aten.linear.default(reshape, p_bert_encoder_layer_0_attention_output_dense_weight, p_bert_encoder_layer_0_attention_output_dense_bias);  reshape = p_bert_encoder_layer_0_attention_output_dense_weight = p_bert_encoder_layer_0_attention_output_dense_bias = None
        dropout_1 = torch.ops.aten.dropout.default(linear_3, 0.1, False);  linear_3 = None
        add_1 = torch.ops.aten.add.Tensor(dropout_1, dropout);  dropout_1 = dropout = None
        layer_norm_1 = torch.ops.aten.layer_norm.default(add_1, [32], p_bert_encoder_layer_0_attention_output_layernorm_weight, p_bert_encoder_layer_0_attention_output_layernorm_bias, 1e-12);  add_1 = p_bert_encoder_layer_0_attention_output_layernorm_weight = p_bert_encoder_layer_0_attention_output_layernorm_bias = None
        linear_4 = torch.ops.aten.linear.default(layer_norm_1, p_bert_encoder_layer_0_intermediate_dense_weight, p_bert_encoder_layer_0_intermediate_dense_bias);  p_bert_encoder_layer_0_intermediate_dense_weight = p_bert_encoder_layer_0_intermediate_dense_bias = None
        gelu = torch.ops.aten.gelu.default(linear_4);  linear_4 = None
        linear_5 = torch.ops.aten.linear.default(gelu, p_bert_encoder_layer_0_output_dense_weight, p_bert_encoder_layer_0_output_dense_bias);  gelu = p_bert_encoder_layer_0_output_dense_weight = p_bert_encoder_layer_0_output_dense_bias = None
        dropout_2 = torch.ops.aten.dropout.default(linear_5, 0.1, False);  linear_5 = None
        add_2 = torch.ops.aten.add.Tensor(dropout_2, layer_norm_1);  dropout_2 = layer_norm_1 = None
        layer_norm_2 = torch.ops.aten.layer_norm.default(add_2, [32], p_bert_encoder_layer_0_output_layernorm_weight, p_bert_encoder_layer_0_output_layernorm_bias, 1e-12);  add_2 = p_bert_encoder_layer_0_output_layernorm_weight = p_bert_encoder_layer_0_output_layernorm_bias = None
        linear_6 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_query_weight, p_bert_encoder_layer_1_attention_self_query_bias);  p_bert_encoder_layer_1_attention_self_query_weight = p_bert_encoder_layer_1_attention_self_query_bias = None
        view_3 = torch.ops.aten.view.default(linear_6, [1, 12, 4, 8]);  linear_6 = None
        permute_3 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        linear_7 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_key_weight, p_bert_encoder_layer_1_attention_self_key_bias);  p_bert_encoder_layer_1_attention_self_key_weight = p_bert_encoder_layer_1_attention_self_key_bias = None
        view_4 = torch.ops.aten.view.default(linear_7, [1, 12, 4, 8]);  linear_7 = None
        permute_4 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        linear_8 = torch.ops.aten.linear.default(layer_norm_2, p_bert_encoder_layer_1_attention_self_value_weight, p_bert_encoder_layer_1_attention_self_value_bias);  p_bert_encoder_layer_1_attention_self_value_weight = p_bert_encoder_layer_1_attention_self_value_bias = None
        view_5 = torch.ops.aten.view.default(linear_8, [1, 12, 4, 8]);  linear_8 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        scaled_dot_product_attention_1 = torch.ops.aten.scaled_dot_product_attention.default(permute_3, permute_4, permute_5, masked_fill);  permute_3 = permute_4 = permute_5 = None
        transpose_1 = torch.ops.aten.transpose.int(scaled_dot_product_attention_1, 1, 2);  scaled_dot_product_attention_1 = None
        reshape_1 = torch.ops.aten.reshape.default(transpose_1, [1, 12, 32]);  transpose_1 = None
        linear_9 = torch.ops.aten.linear.default(reshape_1, p_bert_encoder_layer_1_attention_output_dense_weight, p_bert_encoder_layer_1_attention_output_dense_bias);  reshape_1 = p_bert_encoder_layer_1_attention_output_dense_weight = p_bert_encoder_layer_1_attention_output_dense_bias = None
        dropout_3 = torch.ops.aten.dropout.default(linear_9, 0.1, False);  linear_9 = None
        add_3 = torch.ops.aten.add.Tensor(dropout_3, layer_norm_2);  dropout_3 = layer_norm_2 = None
        layer_norm_3 = torch.ops.aten.layer_norm.default(add_3, [32], p_bert_encoder_layer_1_attention_output_layernorm_weight, p_bert_encoder_layer_1_attention_output_layernorm_bias, 1e-12);  add_3 = p_bert_encoder_layer_1_attention_output_layernorm_weight = p_bert_encoder_layer_1_attention_output_layernorm_bias = None
        linear_10 = torch.ops.aten.linear.default(layer_norm_3, p_bert_encoder_layer_1_intermediate_dense_weight, p_bert_encoder_layer_1_intermediate_dense_bias);  p_bert_encoder_layer_1_intermediate_dense_weight = p_bert_encoder_layer_1_intermediate_dense_bias = None
        gelu_1 = torch.ops.aten.gelu.default(linear_10);  linear_10 = None
        linear_11 = torch.ops.aten.linear.default(gelu_1, p_bert_encoder_layer_1_output_dense_weight, p_bert_encoder_layer_1_output_dense_bias);  gelu_1 = p_bert_encoder_layer_1_output_dense_weight = p_bert_encoder_layer_1_output_dense_bias = None
        dropout_4 = torch.ops.aten.dropout.default(linear_11, 0.1, False);  linear_11 = None
        add_4 = torch.ops.aten.add.Tensor(dropout_4, layer_norm_3);  dropout_4 = layer_norm_3 = None
        layer_norm_4 = torch.ops.aten.layer_norm.default(add_4, [32], p_bert_encoder_layer_1_output_layernorm_weight, p_bert_encoder_layer_1_output_layernorm_bias, 1e-12);  add_4 = p_bert_encoder_layer_1_output_layernorm_weight = p_bert_encoder_layer_1_output_layernorm_bias = None
        linear_12 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_query_weight, p_bert_encoder_layer_2_attention_self_query_bias);  p_bert_encoder_layer_2_attention_self_query_weight = p_bert_encoder_layer_2_attention_self_query_bias = None
        view_6 = torch.ops.aten.view.default(linear_12, [1, 12, 4, 8]);  linear_12 = None
        permute_6 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        linear_13 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_key_weight, p_bert_encoder_layer_2_attention_self_key_bias);  p_bert_encoder_layer_2_attention_self_key_weight = p_bert_encoder_layer_2_attention_self_key_bias = None
        view_7 = torch.ops.aten.view.default(linear_13, [1, 12, 4, 8]);  linear_13 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        linear_14 = torch.ops.aten.linear.default(layer_norm_4, p_bert_encoder_layer_2_attention_self_value_weight, p_bert_encoder_layer_2_attention_self_value_bias);  p_bert_encoder_layer_2_attention_self_value_weight = p_bert_encoder_layer_2_attention_self_value_bias = None
        view_8 = torch.ops.aten.view.default(linear_14, [1, 12, 4, 8]);  linear_14 = None
        permute_8 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        scaled_dot_product_attention_2 = torch.ops.aten.scaled_dot_product_attention.default(permute_6, permute_7, permute_8, masked_fill);  permute_6 = permute_7 = permute_8 = None
        transpose_2 = torch.ops.aten.transpose.int(scaled_dot_product_attention_2, 1, 2);  scaled_dot_product_attention_2 = None
        reshape_2 = torch.ops.aten.reshape.default(transpose_2, [1, 12, 32]);  transpose_2 = None
        linear_15 = torch.ops.aten.linear.default(reshape_2, p_bert_encoder_layer_2_attention_output_dense_weight, p_bert_encoder_layer_2_attention_output_dense_bias);  reshape_2 = p_bert_encoder_layer_2_attention_output_dense_weight = p_bert_encoder_layer_2_attention_output_dense_bias = None
        dropout_5 = torch.ops.aten.dropout.default(linear_15, 0.1, False);  linear_15 = None
        add_5 = torch.ops.aten.add.Tensor(dropout_5, layer_norm_4);  dropout_5 = layer_norm_4 = None
        layer_norm_5 = torch.ops.aten.layer_norm.default(add_5, [32], p_bert_encoder_layer_2_attention_output_layernorm_weight, p_bert_encoder_layer_2_attention_output_layernorm_bias, 1e-12);  add_5 = p_bert_encoder_layer_2_attention_output_layernorm_weight = p_bert_encoder_layer_2_attention_output_layernorm_bias = None
        linear_16 = torch.ops.aten.linear.default(layer_norm_5, p_bert_encoder_layer_2_intermediate_dense_weight, p_bert_encoder_layer_2_intermediate_dense_bias);  p_bert_encoder_layer_2_intermediate_dense_weight = p_bert_encoder_layer_2_intermediate_dense_bias = None
        gelu_2 = torch.ops.aten.gelu.default(linear_16);  linear_16 = None
        linear_17 = torch.ops.aten.linear.default(gelu_2, p_bert_encoder_layer_2_output_dense_weight, p_bert_encoder_layer_2_output_dense_bias);  gelu_2 = p_bert_encoder_layer_2_output_dense_weight = p_bert_encoder_layer_2_output_dense_bias = None
        dropout_6 = torch.ops.aten.dropout.default(linear_17, 0.1, False);  linear_17 = None
        add_6 = torch.ops.aten.add.Tensor(dropout_6, layer_norm_5);  dropout_6 = layer_norm_5 = None
        layer_norm_6 = torch.ops.aten.layer_norm.default(add_6, [32], p_bert_encoder_layer_2_output_layernorm_weight, p_bert_encoder_layer_2_output_layernorm_bias, 1e-12);  add_6 = p_bert_encoder_layer_2_output_layernorm_weight = p_bert_encoder_layer_2_output_layernorm_bias = None
        linear_18 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_query_weight, p_bert_encoder_layer_3_attention_self_query_bias);  p_bert_encoder_layer_3_attention_self_query_weight = p_bert_encoder_layer_3_attention_self_query_bias = None
        view_9 = torch.ops.aten.view.default(linear_18, [1, 12, 4, 8]);  linear_18 = None
        permute_9 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        linear_19 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_key_weight, p_bert_encoder_layer_3_attention_self_key_bias);  p_bert_encoder_layer_3_attention_self_key_weight = p_bert_encoder_layer_3_attention_self_key_bias = None
        view_10 = torch.ops.aten.view.default(linear_19, [1, 12, 4, 8]);  linear_19 = None
        permute_10 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        linear_20 = torch.ops.aten.linear.default(layer_norm_6, p_bert_encoder_layer_3_attention_self_value_weight, p_bert_encoder_layer_3_attention_self_value_bias);  p_bert_encoder_layer_3_attention_self_value_weight = p_bert_encoder_layer_3_attention_self_value_bias = None
        view_11 = torch.ops.aten.view.default(linear_20, [1, 12, 4, 8]);  linear_20 = None
        permute_11 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        scaled_dot_product_attention_3 = torch.ops.aten.scaled_dot_product_attention.default(permute_9, permute_10, permute_11, masked_fill);  permute_9 = permute_10 = permute_11 = None
        transpose_3 = torch.ops.aten.transpose.int(scaled_dot_product_attention_3, 1, 2);  scaled_dot_product_attention_3 = None
        reshape_3 = torch.ops.aten.reshape.default(transpose_3, [1, 12, 32]);  transpose_3 = None
        linear_21 = torch.ops.aten.linear.default(reshape_3, p_bert_encoder_layer_3_attention_output_dense_weight, p_bert_encoder_layer_3_attention_output_dense_bias);  reshape_3 = p_bert_encoder_layer_3_attention_output_dense_weight = p_bert_encoder_layer_3_attention_output_dense_bias = None
        dropout_7 = torch.ops.aten.dropout.default(linear_21, 0.1, False);  linear_21 = None
        add_7 = torch.ops.aten.add.Tensor(dropout_7, layer_norm_6);  dropout_7 = layer_norm_6 = None
        layer_norm_7 = torch.ops.aten.layer_norm.default(add_7, [32], p_bert_encoder_layer_3_attention_output_layernorm_weight, p_bert_encoder_layer_3_attention_output_layernorm_bias, 1e-12);  add_7 = p_bert_encoder_layer_3_attention_output_layernorm_weight = p_bert_encoder_layer_3_attention_output_layernorm_bias = None
        linear_22 = torch.ops.aten.linear.default(layer_norm_7, p_bert_encoder_layer_3_intermediate_dense_weight, p_bert_encoder_layer_3_intermediate_dense_bias);  p_bert_encoder_layer_3_intermediate_dense_weight = p_bert_encoder_layer_3_intermediate_dense_bias = None
        gelu_3 = torch.ops.aten.gelu.default(linear_22);  linear_22 = None
        linear_23 = torch.ops.aten.linear.default(gelu_3, p_bert_encoder_layer_3_output_dense_weight, p_bert_encoder_layer_3_output_dense_bias);  gelu_3 = p_bert_encoder_layer_3_output_dense_weight = p_bert_encoder_layer_3_output_dense_bias = None
        dropout_8 = torch.ops.aten.dropout.default(linear_23, 0.1, False);  linear_23 = None
        add_8 = torch.ops.aten.add.Tensor(dropout_8, layer_norm_7);  dropout_8 = layer_norm_7 = None
        layer_norm_8 = torch.ops.aten.layer_norm.default(add_8, [32], p_bert_encoder_layer_3_output_layernorm_weight, p_bert_encoder_layer_3_output_layernorm_bias, 1e-12);  add_8 = p_bert_encoder_layer_3_output_layernorm_weight = p_bert_encoder_layer_3_output_layernorm_bias = None
        linear_24 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_query_weight, p_bert_encoder_layer_4_attention_self_query_bias);  p_bert_encoder_layer_4_attention_self_query_weight = p_bert_encoder_layer_4_attention_self_query_bias = None
        view_12 = torch.ops.aten.view.default(linear_24, [1, 12, 4, 8]);  linear_24 = None
        permute_12 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        linear_25 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_key_weight, p_bert_encoder_layer_4_attention_self_key_bias);  p_bert_encoder_layer_4_attention_self_key_weight = p_bert_encoder_layer_4_attention_self_key_bias = None
        view_13 = torch.ops.aten.view.default(linear_25, [1, 12, 4, 8]);  linear_25 = None
        permute_13 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        linear_26 = torch.ops.aten.linear.default(layer_norm_8, p_bert_encoder_layer_4_attention_self_value_weight, p_bert_encoder_layer_4_attention_self_value_bias);  p_bert_encoder_layer_4_attention_self_value_weight = p_bert_encoder_layer_4_attention_self_value_bias = None
        view_14 = torch.ops.aten.view.default(linear_26, [1, 12, 4, 8]);  linear_26 = None
        permute_14 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        scaled_dot_product_attention_4 = torch.ops.aten.scaled_dot_product_attention.default(permute_12, permute_13, permute_14, masked_fill);  permute_12 = permute_13 = permute_14 = masked_fill = None
        transpose_4 = torch.ops.aten.transpose.int(scaled_dot_product_attention_4, 1, 2);  scaled_dot_product_attention_4 = None
        reshape_4 = torch.ops.aten.reshape.default(transpose_4, [1, 12, 32]);  transpose_4 = None
        linear_27 = torch.ops.aten.linear.default(reshape_4, p_bert_encoder_layer_4_attention_output_dense_weight, p_bert_encoder_layer_4_attention_output_dense_bias);  reshape_4 = p_bert_encoder_layer_4_attention_output_dense_weight = p_bert_encoder_layer_4_attention_output_dense_bias = None
        dropout_9 = torch.ops.aten.dropout.default(linear_27, 0.1, False);  linear_27 = None
        add_9 = torch.ops.aten.add.Tensor(dropout_9, layer_norm_8);  dropout_9 = layer_norm_8 = None
        layer_norm_9 = torch.ops.aten.layer_norm.default(add_9, [32], p_bert_encoder_layer_4_attention_output_layernorm_weight, p_bert_encoder_layer_4_attention_output_layernorm_bias, 1e-12);  add_9 = p_bert_encoder_layer_4_attention_output_layernorm_weight = p_bert_encoder_layer_4_attention_output_layernorm_bias = None
        linear_28 = torch.ops.aten.linear.default(layer_norm_9, p_bert_encoder_layer_4_intermediate_dense_weight, p_bert_encoder_layer_4_intermediate_dense_bias);  p_bert_encoder_layer_4_intermediate_dense_weight = p_bert_encoder_layer_4_intermediate_dense_bias = None
        gelu_4 = torch.ops.aten.gelu.default(linear_28);  linear_28 = None
        linear_29 = torch.ops.aten.linear.default(gelu_4, p_bert_encoder_layer_4_output_dense_weight, p_bert_encoder_layer_4_output_dense_bias);  gelu_4 = p_bert_encoder_layer_4_output_dense_weight = p_bert_encoder_layer_4_output_dense_bias = None
        dropout_10 = torch.ops.aten.dropout.default(linear_29, 0.1, False);  linear_29 = None
        add_10 = torch.ops.aten.add.Tensor(dropout_10, layer_norm_9);  dropout_10 = layer_norm_9 = None
        layer_norm_10 = torch.ops.aten.layer_norm.default(add_10, [32], p_bert_encoder_layer_4_output_layernorm_weight, p_bert_encoder_layer_4_output_layernorm_bias, 1e-12);  add_10 = p_bert_encoder_layer_4_output_layernorm_weight = p_bert_encoder_layer_4_output_layernorm_bias = None
        slice_5 = torch.ops.aten.slice.Tensor(layer_norm_10, 0, 0, 9223372036854775807);  layer_norm_10 = None
        select = torch.ops.aten.select.int(slice_5, 1, 0);  slice_5 = None
        linear_30 = torch.ops.aten.linear.default(select, p_bert_pooler_dense_weight, p_bert_pooler_dense_bias);  select = p_bert_pooler_dense_weight = p_bert_pooler_dense_bias = None
        tanh = torch.ops.aten.tanh.default(linear_30);  linear_30 = None
        dropout_11 = torch.ops.aten.dropout.default(tanh, 0.1, False);  tanh = None
        linear_31 = torch.ops.aten.linear.default(dropout_11, p_classifier_weight, p_classifier_bias);  dropout_11 = p_classifier_weight = p_classifier_bias = None
        return (linear_31,)
        
    # To see more debug info, please use `graph_module.print_readable()`
