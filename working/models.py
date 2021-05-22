import torch.nn as nn
from transformers import T5ForConditionalGeneration


class mT5Translator(nn.Module):

    def __init__(self, weights, from_tf=True):
        super(mT5Translator, self).__init__()
        self.translator = T5ForConditionalGeneration.from_pretrained(weights, from_tf=True)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        return self.translator(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask)