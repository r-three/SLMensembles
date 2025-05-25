import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(self, model_names, config=None, torch_dtype=torch.bfloat16, device_map="auto", vocab_size=None):
        """
        Args:
            model_names (list): List of Hugging Face model names to ensemble.
        """
        if config is None:
            config = AutoConfig.from_pretrained(model_names[0])
        super().__init__(config)

        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()

        # Load multiple models
        self.models = nn.ModuleList([
            AutoModelForCausalLM.from_pretrained(name, torch_dtype=self.torch_dtype, device_map=self.device_map) for name in model_names
        ])
        
        for model in self.models:
            if self.vocab_size is not None:
                model.resize_token_embeddings(new_num_tokens=self.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Computes the averaged logits over ensemble members.
        """
        logits = None         
        for model in self.models:
            device = model.get_input_embeddings().weight.device
            outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), **kwargs)
            if logits is None:
                logits = outputs.logits.to("cuda:0")
            else:
                logits = logits + outputs.logits.to("cuda:0")
        logits = logits / len(self.models)

        loss = None
        if labels is not None:
            loss = self.models[0].loss_function(logits=logits, labels=labels.to("cuda:0"), vocab_size=self.models[0].config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(logits=logits, loss=loss)
    
    def gradient_checkpointing_enable(self, *args, **kwargs):
        for model in self.models:
            model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        for model in self.models:
            model.gradient_checkpointing_disable(*args, **kwargs)
 
    def add_model(self, model_name):
        """
        Add a new model to the ensemble.
        """
        new_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype, device_map=self.device_map)
        if self.vocab_size is not None:
            new_model.resize_token_embeddings(new_num_tokens=self.vocab_size)
        self.models.append(new_model)

    def remove_model(self, model_idx):
        model = self.models.pop(model_idx)
        del model
