#!/usr/bin/env python3
"""Test script to verify ensemble loading and incremental updates work correctly."""

import torch
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ensemble import ModelEnsemble, EnsembleLoader
from transformers import AutoModelForCausalLM, AutoConfig
import config

def create_mock_checkpoint(output_dir, round_num):
    """Create a mock checkpoint for testing."""
    round_dir = Path(output_dir) / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a small model for testing
    config_obj = AutoConfig.from_pretrained("gpt2")
    config_obj.n_layer = 2  # Reduce layers for faster testing
    config_obj.n_head = 2
    config_obj.n_embd = 128
    
    model = AutoModelForCausalLM.from_config(config_obj)
    
    # Save model state dict
    checkpoint_path = round_dir / "model_state_dict.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    # Also save as hugging face format for testing
    hf_dir = round_dir / "hugging_face"
    hf_dir.mkdir(exist_ok=True)
    model.save_pretrained(hf_dir)
    
    return round_dir

def test_ensemble_loading():
    """Test ensemble loading and incremental updates."""
    print("=" * 60)
    print("Testing Ensemble Loading and Incremental Updates")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override config for testing
        original_model = config.student_model_name
        config.student_model_name = "gpt2"
        config.student_vocab_size = 50257
        
        try:
            # Test 1: Initial load with no models
            print("\n1. Testing initial load with no models...")
            loader = EnsembleLoader(tmpdir)
            ensemble = loader.load_or_update_ensemble(None, device="cpu")
            assert ensemble is None, "Should return None when no models exist"
            print("✓ Initial load with no models works correctly")
            
            # Test 2: Create first model and load
            print("\n2. Testing first model creation and load...")
            round1_dir = create_mock_checkpoint(tmpdir, 0)
            ensemble = loader.load_or_update_ensemble(None, device="cpu")
            assert ensemble is not None, "Should create ensemble with first model"
            assert len(ensemble.models) == 1, "Should have 1 model"
            assert 0 in loader.loaded_rounds or str(round1_dir) in loader.loaded_rounds
            print(f"✓ First model loaded successfully (loaded_rounds: {loader.loaded_rounds})")
            
            # Test 3: Add second model incrementally
            print("\n3. Testing incremental model addition...")
            round2_dir = create_mock_checkpoint(tmpdir, 1)
            ensemble = loader.load_or_update_ensemble(ensemble, device="cpu")
            assert ensemble is not None, "Should return updated ensemble"
            assert len(ensemble.models) == 2, "Should have 2 models"
            print(f"✓ Second model added incrementally (total models: {len(ensemble.models)})")
            
            # Test 4: No duplicate loading
            print("\n4. Testing duplicate load prevention...")
            ensemble_before = ensemble
            ensemble = loader.load_or_update_ensemble(ensemble, device="cpu")
            assert ensemble is ensemble_before, "Should return same ensemble when no new models"
            assert len(ensemble.models) == 2, "Should still have 2 models"
            print("✓ Duplicate loading prevented correctly")
            
            # Test 5: Test add_model method directly
            print("\n5. Testing add_model method...")
            round3_dir = create_mock_checkpoint(tmpdir, 2)
            initial_count = len(ensemble.models)
            ensemble.add_model(round3_dir)
            assert len(ensemble.models) == initial_count + 1, "Should add one model"
            print(f"✓ add_model works correctly (total models: {len(ensemble.models)})")
            
            # Test 6: Test forward pass
            print("\n6. Testing ensemble forward pass...")
            input_ids = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                output = ensemble(input_ids)
            assert output.logits is not None, "Should produce logits"
            assert output.logits.shape[0] == 1, "Batch size should match"
            print(f"✓ Forward pass successful (output shape: {output.logits.shape})")
            
            print("\n" + "=" * 60)
            print("All tests passed successfully! ✨")
            print("=" * 60)
            
        finally:
            # Restore original config
            config.student_model_name = original_model

if __name__ == "__main__":
    test_ensemble_loading()
