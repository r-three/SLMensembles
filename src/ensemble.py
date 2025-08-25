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
            state_dict = torch.load(path / "model_state_dict.pt", weights_only=True, map_location='cpu')
            model.load_state_dict(state_dict)
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

    def add_model(self, model_dir):
        """Add a new model to the ensemble from a saved checkpoint path."""
        model = AutoModelForCausalLM.from_pretrained(self.model_type, torch_dtype=self.torch_dtype)
        model_path = Path(model_dir) / "model_state_dict.pt"
        if model_path.exists():
            try:
                # Always load to CPU first to avoid CUDA memory issues
                state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model state dict: {e}")
                # Fallback to loading from directory
                model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=self.torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=self.torch_dtype)
        
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


class EnsembleLoader:    
    def __init__(self, output_path: str):
        """Class to load ensemble models from completed training rounds."""
        self.ensemble_dir = output_path
        self.model_type = config.student_model_name
        self.loaded_rounds = set()
    
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
       
        if model_rounds:
            ensemble = ModelEnsemble(
                model_paths=model_rounds,
                torch_dtype=torch_dtype,
                vocab_size=config.student_vocab_size,
            ).to(device)
            ensemble.requires_grad_(False)

            self.loaded_rounds.update(model_rounds)
            return ensemble

        return None
    
    def load_or_update_ensemble(self, existing_ensemble, device, torch_dtype=torch.bfloat16):
        """Load initial ensemble or update existing ensemble with new models."""
        model_rounds = self._get_completed_rounds()
        
        if not model_rounds:
            return None
        
        new_rounds = [r for r in model_rounds if r not in self.loaded_rounds]
        
        if existing_ensemble is None and model_rounds:
            ensemble = ModelEnsemble(
                model_paths=model_rounds,
                torch_dtype=torch_dtype,
                vocab_size=config.student_vocab_size,
            ).to(device)
            ensemble.requires_grad_(False)
            self.loaded_rounds.update(model_rounds)
            return ensemble
        elif existing_ensemble and new_rounds:
            for round_path in new_rounds:
                existing_ensemble.add_model(round_path)
            self.loaded_rounds.update(new_rounds)
        
        return existing_ensemble
    
    def save_model_for_ensemble(self, model, round_num: int):
        """Save a trained model."""
        round_dir = os.path.join(self.ensemble_dir, f"round_{round_num}")
        os.makedirs(round_dir)
        
        breakpoint()

        full_model_state_dict = None
        try: 
            safe_state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=False)
            full_model_state_dict = get_model_state_dict(model=model, options=safe_state_dict_opts)
        except Exception as e:
            print(f"Warning: FSDP state dict failed: {e}. Using model.state_dict() fallback...")
            try:
                full_model_state_dict = model.state_dict()
            except Exception as e2:
                print(f"Warning: model.state_dict() also failed: {e2}. Skipping manual state dict save...")
                full_model_state_dict = None
        
        if is_main_process() and full_model_state_dict is not None:
            torch.save(full_model_state_dict, os.path.join(round_dir, "model_state_dict.pt"))

            # try:
            #     hf_dir = os.path.join(round_dir, "hugging_face")
            #     # Unwrap FSDP model before HF save
            #     unwrapped_model = model.module if hasattr(model, 'module') else model
            #     unwrapped_model.save_pretrained(hf_dir, safe_serialization=False)  # Disable safetensors
            # except Exception as e:
            #     print(f"Warning: HF save failed: {e} (not needed for ensemble)")
                    
            print(f"Saved ensemble model for round {round_num} at: {round_dir}")
        
        dist.barrier()
        return round_dir
