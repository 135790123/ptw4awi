import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, T5ForConditionalGeneration, RobertaTokenizer

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, model,config,args):
        super(Model, self).__init__()
        # bert 预训练模型
        self.codet5 = model
        self.tokenizer = RobertaTokenizer.from_pretrained("/root/local_model/codet5-small")
        self.config = config
        for param in self.codet5.parameters():
            param.requires_grad = True  # 使参数可更新
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if output_attentions:
            outputs = self.codet5(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True,output_attentions=output_attentions)
            attentions = outputs.decoder_attentions
            hidden_states = outputs['encoder_last_hidden_state']
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob,attentions
            else:
                return prob,attentions
        else:
            outputs = self.codet5(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=False,
                                  output_attentions=output_attentions)
            hidden_states = outputs['encoder_last_hidden_state']
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob