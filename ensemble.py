import time, csv
import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import config


class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_paths,
        model_base,
        config=None,
        torch_dtype=torch.bfloat16,
        vocab_size=None,
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_base)
        super().__init__(config)

        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_base = model_base

        modules = []
        for path in model_paths:
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=self.torch_dtype)
            model.load_state_dict(torch.load(path / "model_state_dict.pt", weights_only=True))
            model.eval()
            modules.append(model)
        self.models = nn.ModuleList(modules)
        self.models.eval()

        for model in self.models:
            if self.vocab_size is not None:
                model.resize_token_embeddings(new_num_tokens=self.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            outputs = self.models[0](input_ids=input_ids.to(self.models[0].device), attention_mask=attention_mask.to(self.models[0].device), **kwargs)
            sum_logits = outputs.logits
            for model in self.models[1:]:
                sum_logits += model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), **kwargs).logits

            loss = None
            if labels is not None:
                loss = self.models[0].loss_function(
                    logits=outputs.logits,
                    labels=labels.to(outputs.logits.device),
                    vocab_size=config.student_vocab_size,
                    **kwargs
                )

        return CausalLMOutputWithPast(logits=sum_logits, loss=loss)

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
