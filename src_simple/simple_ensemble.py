import torch
import os
from pathlib import Path
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from simple_config import DistillationConfig as config
from simple_checkpoint import AppState

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
        if rank is not None and config.ensemble_dirs:
            num_ensemble_models = len(config.ensemble_dirs)
            has_model = rank < num_ensemble_models
            app = None 

            # only ranks that map to an ensemble model load one
            if has_model:
                model_index = rank  # rank 0 → model 0
                checkpoint_path = config.ensemble_dirs[model_index]
                gpu_id = rank  # rank 0 → cuda:0
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_type, 
                    torch_dtype=self.torch_dtype,
                )
                
                # Load weights
                if os.path.isdir(checkpoint_path):
                    # Distributed checkpoint directory (DCP shards like __0_0.distcp)
                    dummy_optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
                    app = AppState(model=model, optimizer=dummy_optim, lr_scheduler=None)
                elif os.path.isfile(checkpoint_path):
                    # Single-file checkpoint (.pt)
                    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

                    # Handle different saving formats:
                    # 1) checkpoint is already a pure state dict (your final_model/model.pt)
                    # 2) checkpoint is a dict containing "model_state_dict"
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        # Assume it's already a state dict
                        state_dict = checkpoint

                    model.load_state_dict(state_dict, strict=False)

            # ---- DCP LOAD (collective): every rank must enter ----
            if os.path.isdir(config.ensemble_dirs[0]):  # any DCP dir in the list implies DCP loads
                if dist.is_initialized():
                    dist.barrier()

                for i, cp in enumerate(config.ensemble_dirs):
                    # Owner rank i provides {"app": app}; others pass {}
                    state = {"app": app} if (has_model and i == rank) else {}
                    dcp.load(state_dict=state, checkpoint_id=str(cp))

                if dist.is_initialized():
                    dist.barrier()

            # Finalize only on owner ranks
            if has_model:
                if os.path.isdir(checkpoint_path):
                    del dummy_optim

                model = model.to(f"cuda:{gpu_id}")
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
    
