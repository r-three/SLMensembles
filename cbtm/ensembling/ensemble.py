import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from kmeans_pytorch import KMeans as BalancedKMeans

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))



class CBTMInference:
    """
    C-BTM Inference: Sparse ensemble of Expert Language Models
    
    Implements Section 2.2 from "Scaling Expert Language Models with Unsupervised Domain Discovery"
    
    p(X_t | x_{<t}) = Σ_{j=1}^k p(X_t | x_{<t}, D=j) · p(D=j | x_{<t})
    p(D=j | x_{<t}) ∝ topk[exp(-dist(h_{x_{<t}}, h_{c_j})^2 / T)]
    """
    
    def __init__(self, 
                 vectorizer_path: str,
                 cluster_centers_path: str, 
                 expert_paths: List[str],
                 temperature: float = 0.1,
                 top_k: int = None):
        
        # Load clustering components
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(cluster_centers_path, 'rb') as f:
            self.cluster_centers = pickle.load(f)
            
        # Load expert models from checkpoint directories
        self.experts = []
        self.tokenizer = None
        
        print(f"Loading {len(expert_paths)} expert models...")
        
        for i, path in enumerate(expert_paths):
            print(f"Loading expert {i} from {Path(path).name}")
            
            # Load model from checkpoint directory
            expert = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.experts.append(expert)
            
            # Load tokenizer from first expert (assuming all use same tokenizer)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.n_experts = len(self.experts)
        self.temperature = temperature
        self.top_k = top_k if top_k is not None else self.n_experts
        
        # Set all experts to eval mode
        for expert in self.experts:
            expert.eval()
            
        print(f"✅ Loaded {self.n_experts} experts successfully")
    
    def embed_context(self, context: str) -> np.ndarray:
        """Embed context using TF-IDF vectorizer from clustering"""
        embedded = self.vectorizer.transform([context])
        if hasattr(embedded, 'toarray'):
            embedded = embedded.toarray()
        return embedded.astype(np.float32).squeeze()
    
    def compute_ensemble_weights(self, context: str) -> torch.Tensor:
        """
        Compute p(D=j | x_{<t}) using Equation 3:
        p(D=j | x_{<t}) ∝ topk[exp(-dist(h_{x_{<t}}, h_{c_j})^2 / T)]
        """
        # Embed context
        h_context = self.embed_context(context)
        
        # Extract cluster centers from KMeans object
        centers = self.cluster_centers.cluster_centers.cpu().numpy()
        
        # print(f"🔍 Debug: Context embedding shape: {h_context.shape}")
        # print(f"🔍 Debug: Cluster centers shape: {centers.shape}")
            
        # Compute squared distances
        distances = euclidean_distances(h_context.reshape(1, -1), centers).squeeze()
        squared_distances = distances ** 2
        
        # print(f"🔍 Debug: Distances: {distances}")
        # print(f"🔍 Debug: Temperature: {self.temperature}")
        
        # Apply temperature: exp(-dist^2 / T)
        similarities = np.exp(-squared_distances / self.temperature)
        
        # print(f"🔍 Debug: Similarities before top-k: {similarities}")
        
        # Top-k filtering
        if self.top_k < len(similarities):
            top_k_indices = np.argsort(similarities)[-self.top_k:]
            sparse_similarities = np.zeros_like(similarities)
            sparse_similarities[top_k_indices] = similarities[top_k_indices]
            similarities = sparse_similarities
        
        # Normalize to probabilities
        weights = similarities / (np.sum(similarities) + 1e-8)
        
        # print(f"🔍 Debug: Final weights: {weights}")
        # print(f"🔍 Debug: Active experts: {(weights > 1e-8).sum()}")
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def generate_next_token(self, input_ids: torch.Tensor, ensemble_weights: torch.Tensor) -> torch.Tensor:
        """
        Generate next token using ensemble of experts
        
        p(X_t | x_{<t}) = Σ_{j=1}^k p(X_t | x_{<t}, D=j) · p(D=j | x_{<t})
        """
        ensemble_logits = None
        
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                weight = ensemble_weights[i].item()
                
                if weight > 1e-8:  # Only compute for active experts
                    # Get expert logits: p(X_t | x_{<t}, D=j)
                    outputs = expert(input_ids)
                    logits = outputs.logits[:, -1, :]  # Next token logits
                    
                    # Move to CPU for ensemble computation
                    logits = logits.cpu()
                    
                    # Weighted sum
                    if ensemble_logits is None:
                        ensemble_logits = weight * logits
                    else:
                        ensemble_logits += weight * logits
        
        # Handle case where no experts are active
        if ensemble_logits is None:
            print("⚠️ No active experts! Using uniform distribution.")
            vocab_size = self.tokenizer.vocab_size
            ensemble_logits = torch.zeros(1, vocab_size)
        
        return ensemble_logits
    
    # def generate_text(self, 
    #                  context: str, 
    #                  max_new_tokens: int = 50,
    #                  update_weights_every: int = 1,
    #                  do_sample: bool = True,
    #                  sampling_temperature: float = 0.7) -> tuple:
    #     """
    #     Generate text token by token using C-BTM ensemble
    #     """
    #     # Format context as chat message
    #     formatted_context = self._format_chat_prompt(context)
    #     print(f"📝 Formatted prompt: {formatted_context[:200]}...")
        
    #     # Tokenize formatted context and move to device
    #     input_ids = self.tokenizer.encode(formatted_context, return_tensors='pt')
    #     if self.experts:
    #         device = next(self.experts[0].parameters()).device
    #         input_ids = input_ids.to(device)
        
    #     generated_tokens = []
    #     current_context = context  # Use original context for weight computation
        
    #     print(f"🎯 Starting generation...")
        
    #     for step in range(max_new_tokens):
    #         # Recompute ensemble weights periodically (use original context)
    #         if step % update_weights_every == 0:
    #             ensemble_weights = self.compute_ensemble_weights(current_context)
    #             active_experts = (ensemble_weights > 1e-8).sum().item()
    #             if step == 0:
    #                 print(f"🤖 Active experts: {active_experts}/{self.n_experts}")
    #                 for i, w in enumerate(ensemble_weights):
    #                     if w > 1e-8:
    #                         print(f"   Expert {i}: {w:.4f}")
            
    #         # Get next token logits from ensemble
    #         next_token_logits = self.generate_next_token(input_ids, ensemble_weights)
            
    #         # Sample or choose next token
    #         if do_sample:
    #             next_token_probs = F.softmax(next_token_logits / sampling_temperature, dim=-1)
    #             next_token_id = torch.multinomial(next_token_probs, num_samples=1)
    #         else:
    #             next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
    #         # Move to same device as input_ids
    #         next_token_id = next_token_id.to(input_ids.device)
            
    #         # Append to sequence
    #         input_ids = torch.cat([input_ids, next_token_id], dim=1)
    #         generated_tokens.append(next_token_id.item())
            
    #         # Check for stopping conditions
    #         if next_token_id.item() == self.tokenizer.eos_token_id:
    #             print(f"🛑 Generation stopped at EOS token (step {step})")
    #             break
        
    #     # Decode generated text
    #     generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #     full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
    #     print(f"✅ Generated {len(generated_tokens)} tokens")
    #     return generated_text, full_text

    def generate_text(self, 
                    context: str, 
                    max_new_tokens: int = 50,
                    update_weights_every: int = 1,
                    do_sample: bool = True,
                    sampling_temperature: float = 0.7) -> tuple:
        """
        Generate text token by token using C-BTM ensemble
        
        Args:
            update_weights_every: How often to recompute ensemble weights 
                                (1 = every token as per paper, higher = less frequent)
        """
        # Format context as chat message
        formatted_context = self._format_chat_prompt(context)
        print(f"📝 Formatted prompt: {formatted_context[:200]}...")
        
        # Tokenize formatted context and move to device
        input_ids = self.tokenizer.encode(formatted_context, return_tensors='pt')
        if self.experts:
            device = next(self.experts[0].parameters()).device
            input_ids = input_ids.to(device)
        
        generated_tokens = []
        
        print(f"🎯 Starting generation...")
        print(f"📊 Paper mode: Updating weights every {update_weights_every} token(s)")
        
        for step in range(max_new_tokens):
            # Update ensemble weights based on current sequence
            if step % update_weights_every == 0:
                # Decode current sequence for weight computation
                # current_full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True) #TODO uncomment to replicate the main paper
                
                # For weight computation, use the original context (since clustering was done on raw prompts)
                # But we could also experiment with using the full generated text
                ensemble_weights = self.compute_ensemble_weights(formatted_context)
                
                active_experts = (ensemble_weights > 1e-8).sum().item()
                #print(f"🤖 Step {step}: Active experts: {active_experts}/{self.n_experts}")
                
                if step == 0 or step % 10 == 0:  # Show weights initially and every 10 steps
                    for i, w in enumerate(ensemble_weights):
                        if w > 1e-8:
                            print(f"   Expert {i}: {w:.4f}")
            
            # Get next token logits from ensemble
            next_token_logits = self.generate_next_token(input_ids, ensemble_weights)
            
            # Sample or choose next token
            if do_sample:
                next_token_probs = F.softmax(next_token_logits / sampling_temperature, dim=-1)
                next_token_id = torch.multinomial(next_token_probs, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Move to same device as input_ids
            next_token_id = next_token_id.to(input_ids.device)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())
            
            # Decode and print the new token
            new_token = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
            print(f"Token {step}: '{new_token}'", end=" ", flush=True)
            if step % 10 == 9:  # New line every 10 tokens
                print()
            
            # Check for stopping conditions
            if next_token_id.item() == self.tokenizer.eos_token_id:
                print(f"\n🛑 Generation stopped at EOS token (step {step})")
                break
        
        print()  # Final newline
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        print(f"✅ Generated {len(generated_tokens)} tokens")
        return generated_text, full_text

    def _format_chat_prompt(self, context: str) -> str:
        """Format context as a proper chat message using the model's chat template"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Use the model's chat template
            messages = [{"role": "user", "content": context}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted_prompt
            except:
                # Fallback if chat template fails
                pass
        
        # Fallback: simple instruct format
        return f"<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n"
    
    def predict_next_token_probabilities(self, context: str, top_k_tokens: int = 10) -> Dict:
        """
        Get next token probability distribution from ensemble
        """
        # Format context as chat message
        formatted_context = self._format_chat_prompt(context)
        print(f"📝 Formatted prompt: {formatted_context[:200]}...")
        
        # Compute ensemble weights for context (use original context for clustering)
        ensemble_weights = self.compute_ensemble_weights(context)
        
        # Tokenize formatted context and move to device
        input_ids = self.tokenizer.encode(formatted_context, return_tensors='pt')
        if self.experts:
            device = next(self.experts[0].parameters()).device
            input_ids = input_ids.to(device)
        
        # Get ensemble logits
        ensemble_logits = self.generate_next_token(input_ids, ensemble_weights)
        
        # Convert to probabilities
        ensemble_probs = F.softmax(ensemble_logits, dim=-1)
        
        # Get top tokens
        top_probs, top_indices = torch.topk(ensemble_probs, k=top_k_tokens)
        top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices[0]]
        
        return {
            'ensemble_weights': ensemble_weights.cpu().numpy(),
            'top_tokens': list(zip(top_tokens, top_probs[0].cpu().numpy())),
            'active_experts': (ensemble_weights > 1e-8).sum().item(),
            'context': context,
            'formatted_context': formatted_context
        }


def load_cbtm_model(vectorizer_path: str, 
                   cluster_centers_path: str,
                   expert_dir: str,
                   temperature: float = 0.1,
                   top_k: int = None) -> CBTMInference:
    """Load C-BTM model from saved components"""
    
    # Find all expert checkpoint directories
    expert_paths = []
    expert_dir_path = Path(expert_dir)
    
    # Look for numbered directories (checkpoint-*, expert_*, cluster_*, etc.)
    for path in sorted(expert_dir_path.iterdir()):
        if path.is_dir():
            # Check if it contains model files
            if (path / "checkpoint-421/pytorch_model.bin").exists() or (path / "checkpoint-421/model.safetensors").exists():
                expert_paths.append(str(path))
    
    if not expert_paths:
        raise ValueError(f"No expert checkpoint directories found in {expert_dir}")
    
    print(f"Found {len(expert_paths)} expert directories:")
    for i, path in enumerate(expert_paths):
        print(f"  {i}: {Path(path).name}")
    
    return CBTMInference(
        vectorizer_path=vectorizer_path,
        cluster_centers_path=cluster_centers_path,
        expert_paths=expert_paths,
        temperature=temperature,
        top_k=top_k
    )


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorizer', required=True, help='Path to vectorizer.pkl')
    parser.add_argument('--cluster-centers', required=True, help='Path to cluster_centers.pkl')
    parser.add_argument('--expert-dir', required=True, help='Directory with expert checkpoint folders')
    parser.add_argument('--context', required=True, help='Input context')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for ensemble weights')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k experts to use')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--generate', action='store_true', help='Generate text (vs just show weights)')
    
    args = parser.parse_args()
    
    # Load model
    print("🚀 Loading C-BTM model...")
    cbtm = load_cbtm_model(
        vectorizer_path=args.vectorizer,
        cluster_centers_path=args.cluster_centers,
        expert_dir=args.expert_dir,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(f"\n📝 Context: {args.context}")
    print(f"🌡️ Temperature: {args.temperature}")
    print(f"🔝 Top-K: {cbtm.top_k}")
    
    if args.generate and args.max_tokens > 0:
        # Generate text
        print(f"\n🎯 Generating {args.max_tokens} tokens...")
        generated_text, full_text = cbtm.generate_text(
            context=args.context,
            max_new_tokens=args.max_tokens
        )
        
        print(f"\n📄 Generated text:")
        print(f"'{generated_text}'")
        print(f"\n📖 Full text:")
        print(f"'{full_text}'")
        
    else:
        # Just show ensemble weights and next token probabilities
        result = cbtm.predict_next_token_probabilities(args.context)
        
        print(f"\n⚖️ Ensemble weights:")
        for i, w in enumerate(result['ensemble_weights']):
            if w > 1e-8:
                print(f"  Expert {i}: {w:.4f}")
        
        print(f"\n🎯 Top next tokens:")
        for token, prob in result['top_tokens'][:5]:
            print(f"  '{token}': {prob:.4f}")
        
        print(f"\n🤖 Active experts: {result['active_experts']}/{cbtm.n_experts}")


if __name__ == "__main__":
    main()