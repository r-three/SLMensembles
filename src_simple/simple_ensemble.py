import torch
import os
from pathlib import Path
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from simple_config import DistillationConfig as config

class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_type=config.student_model_name,
        torch_dtype=torch.bfloat16,
        vocab_size=None,
        rank=None,
        starting_gpu_index=1,
    ):
        super().__init__(AutoConfig.from_pretrained(model_type))
        
        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_type = model_type

        modules = []
        if rank is not None and rank > 0 and config.ensemble_dirs:
            num_ensemble_models = len(config.ensemble_dirs)
            if rank <= num_ensemble_models:
                model_index = rank - 1  # rank 1 → model 0
                checkpoint_path = config.ensemble_dirs[model_index]
                gpu_id = rank  # rank 1 → cuda:1
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_type, 
                    torch_dtype=self.torch_dtype,
                )
                
                checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
                state_dict = checkpoint['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
                
                model = model.to(f'cuda:{gpu_id}')
                
                model.eval()
                model.requires_grad_(False)
                modules.append(model)
        
        self.models = nn.ModuleList(modules)


    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            first_model = self.models[0]
            outputs = first_model(
                input_ids=input_ids.to(first_model.device), 
                attention_mask=attention_mask.to(first_model.device), 
                **kwargs
            )
            sum_logits = outputs.logits.clone()
            
            for model in self.models[1:]:
                model_outputs = model(
                    input_ids=input_ids.to(model.device), 
                    attention_mask=attention_mask.to(model.device), 
                    **kwargs
                )
                sum_logits += model_outputs.logits.to(sum_logits.device)
            
            logits = sum_logits / len(self.models)
            
            loss = None
            if labels is not None:
                loss = first_model.loss_function(
                    logits=logits,
                    labels=labels.to(logits.device),
                    vocab_size=config.student_vocab_size,
                    **kwargs
                )
        return CausalLMOutputWithPast(logits=logits, loss=loss)

    def add_model(self, model_dir):
        """Add a new model to the ensemble from a saved checkpoint path."""
        model = AutoModelForCausalLM.from_pretrained(self.model_type, torch_dtype=self.torch_dtype)
        
        # Check if it's a checkpoint file (.pt) or a directory with saved model
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        if model_path.suffix == '.pt':
            checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
        elif model_path.is_dir():
            model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=self.torch_dtype)
        else:
            raise ValueError(f"Cannot load model from {model_path}")
        
        model.eval()
        model.requires_grad_(False)
        
        if self.models:
            model = model.to(self.models[0].device)
        
        self.models.append(model)

    def remove_model(self, model_idx):
        if model_idx < len(self.models):
            model = self.models.pop(model_idx)
            del model
            torch.cuda.empty_cache()
    
