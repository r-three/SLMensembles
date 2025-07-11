import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import config


class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_names,
        config=None,
        torch_dtype=torch.bfloat16,
        vocab_size=None,
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_names[0])
        super().__init__(config)

        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()

        self.models = nn.ModuleList([AutoModelForCausalLM.from_pretrained(name, torch_dtype=self.torch_dtype) for name in model_names])

        for model in self.models:
            if self.vocab_size is not None:
                model.resize_token_embeddings(new_num_tokens=self.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        all_logits = []
        for model in self.models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            all_logits.append(outputs.logits)
        logits = torch.stack(all_logits, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            loss = self.models[0].loss_function(
                logits=logits,
                labels=labels.to(logits.device),
                vocab_size=config.student_vocab_size,
                **kwargs
            )
        return CausalLMOutputWithPast(logits=logits, loss=loss)

    def add_model(self, model_name):
        new_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        if self.vocab_size is not None:
            new_model.resize_token_embeddings(new_num_tokens=self.vocab_size)
        self.models.append(new_model)

    def remove_model(self, model_idx):
        if model_idx < len(self.models):
            model = self.models.pop(model_idx)
            del model
            torch.cuda.empty_cache()
