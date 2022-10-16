from torch import nn
import torch
import math
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


def init_bert_weights(module):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Adapter, self).__init__()

        self.down = torch.nn.Linear(input_dim, hidden_dim)
        self.non_linear_func = torch.nn.ReLU()
        self.up = torch.nn.Linear(hidden_dim, input_dim)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        output = self.down(x)
        output = self.non_linear_func(output)
        output = self.up(output)
        return x + output


class classifier(nn.Module):
    def __init__(self, input_dim, labels=2):
        super(classifier, self).__init__()
        self.up = torch.nn.Linear(input_dim, labels)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        output = self.up(x)
        return output


def adpater_bert_forward(adapter, bert, input_ids, attention_mask):
    input_shape = input_ids.size()
    batch_size, seq_length = input_shape

    buffered_token_type_ids = bert.embeddings.token_type_ids[:, :seq_length]
    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    token_type_ids = buffered_token_type_ids_expanded

    extended_attention_mask: torch.Tensor = bert.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
    encoder_extended_attention_mask = None
    head_mask = bert.get_head_mask(None, bert.config.num_hidden_layers)

    embedding_output = bert.embeddings(
        input_ids=input_ids,
        position_ids=None,
        token_type_ids=token_type_ids,
        inputs_embeds=None,
        past_key_values_length=0,
    )

    # adapter
    embedding_output = adapter(embedding_output)

    encoder_outputs = bert.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=(bert.config.output_hidden_states),
        return_dict=True,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = bert.pooler(sequence_output) if bert.pooler is not None else None

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )
