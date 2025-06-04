import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, GPT2Tokenizer

        
class Model(torch.nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer =GPT2Tokenizer.from_pretrained('gpt2')
        self.config = config
        self.classifier = nn.Linear(config.n_embd, 2, bias=False)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.transformer(input_ids=input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.transformer(inputs_embeds=input_embed, output_attentions=output_attentions)
            hidden_states = outputs[0]
            logits = self.classifier(hidden_states)
            attention = outputs.attentions
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
            else:
                batch_size, sequence_length = input_embed.shape[:2]
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            prob = torch.softmax(pooled_logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
                return loss, prob,attention
            else:
                return prob,attention
        else:
            if input_ids is not None:
                outputs = self.encoder.transformer(input_ids=input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.transformer(inputs_embeds=input_embed, output_attentions=output_attentions)
            hidden_states = outputs[0]
            logits = self.classifier(hidden_states)
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape[:2]
            else:
                batch_size, sequence_length = input_embed.shape[:2]
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            prob = torch.softmax(pooled_logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, 2), labels.view(-1))
                return loss, prob
            else:
                return prob