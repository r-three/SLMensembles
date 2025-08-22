import torch
import os
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import config


class ModelEnsemble(PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        model_paths,
        model_type,
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
    def __init__(self, output_path: str, model_type: str, round_num: int):
        """Class to load ensemble models from completed training rounds."""
        self.output_path = output_path
        self.model_type = model_type
        self.round_num = round_num
        os.makedirs(self.ensemble_dir, exist_ok=True)
    
    def save_model_for_ensemble(self, model, round_num: int):
        """Save a trained model."""
        # NOTE: can load model with Can load with torch.load() and use directly

        
        # Create round-specific directory
        round_dir = os.path.join(self.output_path, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # Get full model state dict (gathered on CPU for rank 0 to save)
        full_state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        
        try:
            # For FSDP models, use distributed checkpoint API
            full_model_state_dict = get_model_state_dict(model=model, options=full_state_dict_opts)
        except:
            # Fallback for non-FSDP models
            full_model_state_dict = model.state_dict()
        
        # Only rank 0 saves the model files
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Save model state dict (expected by ModelEnsemble)
            torch.save(full_model_state_dict, os.path.join(round_dir, "model_state_dict.pt"))
            
            # Also save in HuggingFace format for compatibility
            try:
                # Try to save as HuggingFace model if possible
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(round_dir)
                elif hasattr(model, 'module') and hasattr(model.module, 'save_pretrained'):
                    # Handle wrapped models
                    model.module.save_pretrained(round_dir)
                else:
                    # Create a minimal config and save manually
                    from transformers import AutoConfig
                    config_obj = AutoConfig.from_pretrained(self.model_type)
                    config_obj.save_pretrained(round_dir)
            except Exception as e:
                print(f"Warning: Could not save HuggingFace format: {e}")
            
            print(f"Saved ensemble model for round {round_num} at: {round_dir}")
        
        # Ensure all ranks wait for the save to complete
        if dist.is_initialized():
            dist.barrier()
        
        return round_dir
    
    def get_completed_rounds(self):
        """Get list of completed rounds by scanning ensemble model directory."""
        completed_rounds = []
        
        if not os.path.exists(self.ensemble_dir):
            return completed_rounds
            
        for dir_name in os.listdir(self.ensemble_dir):
            if dir_name.startswith('round_'):
                try:
                    round_num = int(dir_name.split('_')[1])
                    round_dir = os.path.join(self.ensemble_dir, dir_name)
                    # Check if the model file exists
                    model_file = os.path.join(round_dir, "model_state_dict.pt")
                    if os.path.isfile(model_file):
                        completed_rounds.append(round_num)
                except (ValueError, IndexError):
                    continue
        
        return sorted(completed_rounds)
    
    def get_model_paths_for_ensemble(self, max_round: int = None):
        """
        Get paths to saved ensemble models up to max_round.
        
        Args:
            max_round: Maximum round to include (exclusive). If None, include all.
            
        Returns:
            list: Paths to model directories for ensemble creation
        """
        completed_rounds = self.get_completed_rounds()
        
        if max_round is not None:
            completed_rounds = [r for r in completed_rounds if r < max_round]
        
        model_paths = []
        for round_num in completed_rounds:
            round_dir = os.path.join(self.ensemble_dir, f"round_{round_num}")
            if os.path.isdir(round_dir):
                model_paths.append(round_dir)
        
        return model_paths
    
    def _has_model_checkpoint(self, round_dir):
        """Check if a round directory has a model checkpoint."""
        for item in os.listdir(round_dir):
            step_dir = os.path.join(round_dir, item)
            if os.path.isdir(step_dir) and item.startswith('step_'):
                model_file = os.path.join(step_dir, "model_state_dict.pt")
                if os.path.exists(model_file):
                    return True
        return False
    
    def load_ensemble_for_round(self, target_round, device, torch_dtype=torch.bfloat16, vocab_size=None):
        """Load ensemble of models from all completed rounds before target_round."""
        completed_rounds = self.get_completed_rounds()
        
        # Only use rounds before the target round
        ensemble_rounds = [r for r in completed_rounds if r < target_round]
        
        if not ensemble_rounds:
            return None  # No ensemble models available
        
        # Get best checkpoint paths for each round
        model_paths = []
        for round_num in ensemble_rounds:
            checkpoint_path = self.get_best_checkpoint_for_round(round_num)
            if checkpoint_path:
                model_paths.append(checkpoint_path)
        
        if not model_paths:
            return None
        
        # Create ensemble
        ensemble = ModelEnsemble(
            model_paths=model_paths,
            model_type=config.student_model_name,
            torch_dtype=torch_dtype,
            vocab_size=vocab_size
        ).to(device)
        ensemble.requires_grad_(False)
        
        return ensemble
