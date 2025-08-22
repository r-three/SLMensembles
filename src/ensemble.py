import torch
import os
from torch import nn
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
    def __init__(self, output_path: str, model_base: str):
        """Class to load ensemble models from completed training rounds."""
        self.output_path = output_path
    
    def get_completed_rounds(self):
        """Get list of completed rounds by scanning checkpoint directory."""
        
        model_dirs = []
        
        for dir_name in os.listdir(self.output_path):
            if dir_name.startswith('round_'):
                round_dir = os.path.join(self.output_path, dir_name)
                model_dirs.append(round_dir)
        
        return sorted(model_dirs)


    # TODO: stopped here


        if existing_models:
            existing_models.sort(key=lambda x: x[0])
            start_round = max((r for r, _ in existing_models)) + 1
            ensemble_model_names = [path for _, path in existing_models]
            ensemble_model = ModelEnsemble(
                model_names=ensemble_model_names,
                torch_dtype=torch.bfloat16,
                device_map=config.student_device,
                vocab_size=student_model.vocab_size,
            )
            ensemble_model.requires_grad_(False)

        completed_rounds = []
        for item in os.listdir(self.checkpoints_dir):
            try:
                round_num = int(item)
                round_dir = os.path.join(self.checkpoints_dir, item)
                if os.path.isdir(round_dir) and self._has_model_checkpoint(round_dir):
                    completed_rounds.append(round_num)
            except ValueError:
                continue
        
        return sorted(completed_rounds)
    
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
            model_base=config.student_model_name,
            torch_dtype=torch_dtype,
            vocab_size=vocab_size
        ).to(device)
        ensemble.requires_grad_(False)
        
        return ensemble
