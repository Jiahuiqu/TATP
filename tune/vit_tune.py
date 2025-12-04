from typing import Union, Tuple, Optional

import math
from transformers import ViTModel
import torch
import torch.nn as nn

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x


class Forward_prompt(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_modalities):
        super(Forward_prompt, self).__init__()
        self.num_modalities = num_modalities
        self.down_projects = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_modalities)])
        self.activations = nn.ModuleList([nn.GELU() for _ in range(num_modalities)])
        self.up_projects = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(num_modalities)])

    def forward(self, prompts):
        processed_prompts = []

        for i, prompt in enumerate(prompts):
            prompt = prompt.view(prompt.size(0), prompt.size(1), -1)
            x = self.down_projects[i](prompt)
            x = self.activations[i](x)
            x = self.up_projects[i](x)
            processed_prompts.append(x)
        return tuple(processed_prompts)


class Attention_KV(nn.Module):
    def __init__(self, Layer):
        super().__init__()
        self.query = Layer.attention.query
        self.key = Layer.attention.key
        self.value = Layer.attention.value
        self.num_attention_heads = Layer.attention.num_attention_heads
        self.attention_head_size = Layer.attention.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self, hidden_states_pro, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        hidden_states = hidden_states_pro[0]
        prompt_embeddings = hidden_states_pro[1:]

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Modality-Special Control - prompt
        for prompt_embedding in prompt_embeddings:
            key = self.transpose_for_scores(self.key(prompt_embedding))
            value = self.transpose_for_scores(self.value(prompt_embedding))
            context_layer_prompt = scaled_dot_product_attention(
                query_layer,
                key,
                value,
            )
            context_layer_prompt = context_layer_prompt.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_prompt.size()[:-2] + (self.all_head_size,)
            context_layer_prompt = context_layer_prompt.view(new_context_layer_shape)
            context_layer = torch.add(context_layer, context_layer_prompt)

        return context_layer, None


class ViTLayer2(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, Layer) -> None:
        super().__init__()
        self.attention = Layer.attention
        # self.attention = Attention_KV(Layer.attention.attention)
        self.intermediate = Layer.intermediate
        self.output = Layer.output
        self.layernorm_before = Layer.layernorm_before
        self.layernorm_after = Layer.layernorm_after

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, Layer,device, config, use_adapter=True) -> None:
        super().__init__()
        self.use_adapter = use_adapter
        self.attention = Layer.attention
        self.attention.attention = Attention_KV(Layer.attention)
        self.intermediate = Layer.intermediate
        self.output = Layer.output
        self.layernorm_before = Layer.layernorm_before
        self.layernorm_after = Layer.layernorm_after
        self.device = device

        if self.use_adapter:
            self.adapters = Adapter(config.hidden_size, 64, config.intermediate_size).to(self.device)
        else:
            self.adapters = None

    def forward(
            self,
            hidden_states_pro,
    ):
        hidden_states = hidden_states_pro[0]
        prompt_embeddings = hidden_states_pro[1:]
        layernorm_before = self.layernorm_before(hidden_states)
        union = (layernorm_before,)
        for prompt_layernorm in prompt_embeddings:
            prompt_layernorm = self.layernorm_before(prompt_layernorm)
            union = union + (prompt_layernorm,)

        self_attention_outputs = self.attention(
            union, head_mask=None, output_attentions=False  # in ViT, layernorm is applied before self-attention
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output_ = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output_)
        if self.use_adapter:
        # add adapter
            layer_output_adapter = self.adapters(layer_output_).to(self.device)
            layer_output = torch.add(layer_output, layer_output_adapter)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class Vit_encoder(nn.Module):
    def __init__(self, vit_model,device, adapter_hidden_dim=64, use_adapter=True, len_modility=2):
        super().__init__()
        self.vit_model = vit_model
        for param in self.vit_model.parameters():
            param.requires_grad = False
        self.vit_config = vit_model.config
        self.encoder = self.vit_model.encoder
        self.adapter_hidden_dim = adapter_hidden_dim
        self.use_adapter = use_adapter
        self.len_modility = len_modility
        self.device = device


        self.forward_prompt = nn.ModuleList([
            Forward_prompt(vit_model.config.hidden_size, 64, vit_model.config.hidden_size, len_modility)
            for _ in range(vit_model.config.num_hidden_layers)
        ])

    def forward(self, embedding, prompt_embeddings):
        numLayers = self.vit_config.num_hidden_layers
        output = ((embedding,) + prompt_embeddings)
        # output = ((embedding,))

        for i in range(numLayers):
            layer = ViTLayer(self.encoder.layer[i],self.device, config=self.vit_config, use_adapter=self.use_adapter)
            output = layer((output[0],)+prompt_embeddings )
            prompt_embeddings = self.forward_prompt[i](prompt_embeddings)
        return output[0]


def compare_models_vectors(model1, model2):
    vec1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
    vec2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
    return torch.allclose(vec1, vec2)


if __name__ == '__main__':
    vit = ViTModel.from_pretrained('..//large_vit_huge')
    config = vit.config

    vit2 = ViTModel.from_pretrained('..//large_vit_huge')
    batch_size = 1
    seq_len = 16
    embedding_dim = 1024
    input_embedding = torch.rand(batch_size, seq_len, embedding_dim)
    input_embedding2 = torch.rand(batch_size, seq_len, embedding_dim)
    prompt_embedding = torch.rand(batch_size, seq_len, embedding_dim)

    prompt_embeddings = (prompt_embedding, prompt_embedding)

    outputs = (input_embedding,)
    outputs2 = (input_embedding2, prompt_embedding)
    outputs3 = outputs

    for i in range(12):
        layer = ViTLayer(vit.encoder.layer[i],device='cpu', config=config, use_adapter=True)
        layer2 = ViTLayer2(vit2.encoder.layer[i])
        outputs = layer((outputs[0],) + prompt_embeddings)
        outputs2 = layer2(outputs2[0])
    encode = Vit_encoder(vit,device='cpu', use_adapter=True, adapter_hidden_dim=64, len_modility=2)
    outputs3 = encode(outputs[0], prompt_embeddings)

    print(vit.layernorm(outputs[0]))
    print(vit.layernorm(outputs3[0]))
