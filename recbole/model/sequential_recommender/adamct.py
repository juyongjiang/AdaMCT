# -*- coding: utf-8 -*-
import torch
import copy
import math
import numpy as np
from torch import nn
import torch.nn.functional as fn

from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.layers import AdaMCTEncoder
from recbole.model.loss import BPRLoss


class AdaMCT(SequentialRecommender):
    def __init__(self, config, dataset):
        super(AdaMCT, self).__init__(config, dataset)
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.kernel_size = config["kernel_size"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"], # max sequence length
        self.reduction_ratio = config["reduction_ratio"],
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.adamct_encoder = AdaMCTEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            seq_len = self.seq_len, 
            reduction_ratio = self.reduction_ratio,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        adamct_output = self.adamct_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = adamct_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class AdaMCTEncoder(nn.Module):
    r"""One AdaMCTEncoder consists of several AdaMCTLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        kernel_size(num): the size of kernel in convolutional layer. Default: 2
        reduction_ratio(num): the reduction rate in squeeze-excitation attention layer. Default: 2
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in convolutional layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        kernel_size=3,
        seq_len=64, 
        reduction_ratio=2,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(AdaMCTEncoder, self).__init__()
        layer = AdaMCTLayer(
            n_heads,
            hidden_size,
            kernel_size,
            seq_len, 
            reduction_ratio,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class AdaMCTLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        kernel_size,
        seq_len, 
        reduction_ratio,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(AdaMCTLayer, self).__init__()
        self.linear_en = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.multi_head_attention = MultiHeadAttention(n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.local_conv = LocalConv(
            hidden_size,
            kernel_size, 
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.global_seatt = SqueezeExcitationAttention(seq_len, reduction_ratio)
        self.local_seatt = SqueezeExcitationAttention(seq_len, reduction_ratio)

        self.adaptive_mixture_units = AdaptiveMixtureUnits(hidden_size, layer_norm_eps, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        hidden_states_en = self.LayerNorm(self.dropout(self.linear_en(hidden_states)))

        global_output = self.multi_head_attention(hidden_states_en, attention_mask)
        global_output = self.global_seatt(global_output)

        local_output = self.local_conv(hidden_states_en)
        local_output = self.local_seatt(local_output)

        layer_output = self.adaptive_mixture_units(hidden_states, global_output, local_output)
        return layer_output


class AdaptiveMixtureUnits(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super(AdaptiveMixtureUnits, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.adaptive_act_fn = torch.sigmoid
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor, global_output, local_output):
        input_tensor_avg = torch.mean(input_tensor, dim=1) # [B, D]
        ada_score_alpha = self.adaptive_act_fn(self.linear(input_tensor_avg)).unsqueeze(-1) # [B, 1, 1]
        ada_score_beta = 1 - ada_score_alpha
        
        mixture_output = torch.mul(global_output, ada_score_beta) + torch.mul(local_output, ada_score_alpha) # [B, N, D]

        output = self.LayerNorm(self.dropout(self.linear_out(mixture_output)) + input_tensor) # [B, N, D]
        return output


class SqueezeExcitationAttention(nn.Module):
    def __init__(self, seq_len, reduction_ratio):
        super(SqueezeExcitationAttention, self).__init__()
        self.dense_1 = nn.Linear(seq_len[0], seq_len[0]//reduction_ratio[0])
        self.squeeze_act_fn = fn.relu

        self.dense_2 = nn.Linear(seq_len[0]//reduction_ratio[0], seq_len[0])
        self.excitation_act_fn = torch.sigmoid

    def forward(self, input_tensor):
        input_tensor_avg = torch.mean(input_tensor, dim=-1, keepdim=True) # [B, N, 1]
        
        hidden_states = self.dense_1(input_tensor_avg.permute(0, 2, 1)) # [B, 1, N] -> [B, 1, N/r]
        hidden_states = self.squeeze_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states) # [B, 1, N/r] -> [B, 1, N] 
        att_score = self.excitation_act_fn(hidden_states) # sigmoid
        
        # reweight
        input_tensor = torch.mul(input_tensor, att_score.permute(0, 2, 1)) # [B, N, D]
        return input_tensor


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class LocalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(LocalConv, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv_1_3 = nn.Conv1d(hidden_size, hidden_size, kernel_size, stride=1, padding=self.padding)
        self.conv_act_fn = self.get_hidden_act(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        # self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.conv_1_3(input_tensor.permute(0, 2, 1))
        hidden_states = self.LayerNorm(hidden_states.permute(0, 2, 1))
        hidden_states = self.conv_act_fn(hidden_states)
        return hidden_states