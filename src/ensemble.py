import torch
import os
from pathlib import Path
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import is_main_process
import config


class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_paths,
        model_type=config.student_model_name,
        config=None,
        torch_dtype=torch.bfloat16,
        vocab_size=None,
    ):
        if config is None:
            config = AutoConfig.from_pretrained(model_type)
        super().__init__(config)

        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.model_type = model_type

        modules = []
        for path in model_paths:
            model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=self.torch_dtype)
            # TODO: loading on device or on CPU
            model.load_state_dict(torch.load(path / "model_state_dict.pt", weights_only=True))
            model.eval()
            model.requires_grad_(False)
            modules.append(model)
        self.models = nn.ModuleList(modules)
        self.models.eval()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            outputs = self.models[0](input_ids=input_ids.to(self.models[0].device), attention_mask=attention_mask.to(self.models[0].device), **kwargs)
            sum_logits = outputs.logits
            for model in self.models[1:]:
                sum_logits += model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), **kwargs).logits

            logits = sum_logits / len(self.models)
            loss = None
            if labels is not None:
                loss = self.models[0].loss_function(
                    logits=outputs.logits,
                    labels=labels.to(outputs.logits.device),
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


class EnsembleLoader:    
    def __init__(self, output_path: str):
        """Class to load ensemble models from completed training rounds."""
        self.ensemble_dir = output_path
        self.model_type = config.student_model_name
    
    def _get_completed_rounds(self):
        """Get list of completed rounds by scanning ensemble model directory."""
        completed_rounds = []
        
        if not self.ensemble_dir.exists():
            return completed_rounds
            
        for path in self.ensemble_dir.iterdir():
            if path.name.startswith('round_'):
                try:
                    round_dir = os.path.join(self.ensemble_dir, path)
                    model_file = os.path.join(round_dir, "model_state_dict.pt")
                    if os.path.isfile(model_file):
                        completed_rounds.append(model_file)
                except (ValueError, IndexError):
                    continue
        
        return sorted(completed_rounds)
    
    def load_ensemble(self, device, torch_dtype=torch.bfloat16):
        """Load and create ensemble of models from completed rounds for use in training."""
        model_rounds = self._get_completed_rounds()
        
        if not model_rounds:
            return None
        
        ensemble = ModelEnsemble(
            model_paths=model_rounds,
            torch_dtype=torch_dtype,
            vocab_size=config.vocab_size,
        ).to(device)
        ensemble.requires_grad_(False)

        return ensemble
    
    def save_model_for_ensemble(self, model, round_num: int):
        """Save a trained model."""
        round_dir = os.path.join(self.output_path, f"round_{round_num}")
        hf_dir = os.path.join(round_dir, "hugging_face")
        os.makedirs(round_dir)
        
        full_state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        
        try: # For FSDP models
            full_model_state_dict = get_model_state_dict(model=model, options=full_state_dict_opts)
        except: # Fallback for non-FSDP models
            full_model_state_dict = model.state_dict()
        
        if is_main_process() == 0:
            torch.save(full_model_state_dict, os.path.join(round_dir, "model_state_dict.pt"))

             # Try to save as HuggingFace
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(hf_dir)
            elif hasattr(model, 'module') and hasattr(model.module, 'save_pretrained'):
                model.module.save_pretrained(hf_dir)
            else:
                print(f"Warning: Could not save model in HuggingFace format")
            
            print(f"Saved ensemble model for round {round_num} at: {round_dir}")
        
        dist.barrier()
        return round_dir
