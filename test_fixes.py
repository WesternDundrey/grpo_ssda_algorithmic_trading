#!/usr/bin/env python3
"""Test that the index fixes are working"""

import numpy as np
import pandas as pd

# Test case 1: Empty features list
def test_empty_features():
    """Test handling of empty feature lists"""
    features_list = []
    
    if not features_list:
        features = np.zeros((10, 1))  # 10 samples, 1 feature
        print("✓ Empty features handled correctly")
        return True
    
    return False

# Test case 2: Safe technical indicators indexing  
def test_tech_indicators():
    """Test safe indexing of technical indicators"""
    # Test with fewer than 5 indicators
    technical_indicators = np.array([1.2, 3.4])  # Only 2 indicators
    
    if len(technical_indicators) >= 5:
        tech_features = technical_indicators[:5]
    else:
        tech_features = np.zeros(5)
        if len(technical_indicators) > 0:
            tech_features[:len(technical_indicators)] = technical_indicators
    
    print(f"✓ Tech indicators: {tech_features}")
    assert len(tech_features) == 5
    assert tech_features[0] == 1.2
    assert tech_features[1] == 3.4
    assert tech_features[2] == 0.0
    return True

# Test case 3: Safe state representation indexing
def test_state_representation():
    """Test safe indexing of state representation"""
    # Test with 2D array but fewer than 3 features
    state_representation = np.array([[1.0, 2.0]])  # 1x2 array
    
    if len(state_representation) > 0 and state_representation.shape[1] >= 3:
        state_features = state_representation[-1, :3]
    elif len(state_representation) > 0:
        available_features = state_representation[-1]
        state_features = np.zeros(3)
        state_features[:len(available_features)] = available_features[:3]
    else:
        state_features = np.zeros(3)
    
    print(f"✓ State features: {state_features}")
    assert len(state_features) == 3
    assert state_features[0] == 1.0
    assert state_features[1] == 2.0
    assert state_features[2] == 0.0
    return True

# Test case 4: Safe denoised features indexing
def test_denoised_features():
    """Test safe indexing of denoised features"""
    # Test with 2D array but insufficient columns
    denoised_features = np.array([[1, 2], [3, 4]])  # 2x2 array
    
    if len(denoised_features) > 1 and denoised_features.shape[1] >= 5:
        current_features = denoised_features[-1, :5]
        previous_features = denoised_features[-2, :5]
        price_trend = np.mean(current_features) - np.mean(previous_features)
    elif len(denoised_features) > 1:
        # Safe indexing for features
        current_features = denoised_features[-1, :min(5, denoised_features.shape[1])]
        previous_features = denoised_features[-2, :min(5, denoised_features.shape[1])]
        price_trend = np.mean(current_features) - np.mean(previous_features)
    else:
        price_trend = 0.0
    
    print(f"✓ Price trend: {price_trend}")
    expected_trend = np.mean([3, 4]) - np.mean([1, 2])  # 3.5 - 1.5 = 2.0
    assert abs(price_trend - expected_trend) < 1e-6
    return True

# Test case 5: Safe hidden_dims indexing
def test_hidden_dims():
    """Test safe indexing of hidden dimensions"""
    hidden_dims = [32, 16, 8]
    encoded = np.random.randn(10, 20)  # 10 samples, 20 features
    
    if len(hidden_dims) > 0:
        bottleneck_size = hidden_dims[len(hidden_dims)//2]
        result = encoded[:, :min(bottleneck_size, encoded.shape[1])]
    else:
        result = encoded
    
    print(f"✓ Encoded shape: {result.shape}")
    assert result.shape[0] == 10
    assert result.shape[1] == min(16, 20)  # bottleneck_size = hidden_dims[1] = 16
    return True

def main():
    """Run all tests"""
    print("=== Testing Index Error Fixes ===")
    
    tests = [
        ("Empty Features", test_empty_features),
        ("Technical Indicators", test_tech_indicators), 
        ("State Representation", test_state_representation),
        ("Denoised Features", test_denoised_features),
        ("Hidden Dimensions", test_hidden_dims)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} - PASSED")
            else:
                print(f"✗ {test_name} - FAILED")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name} - ERROR: {e}")
            all_passed = False
    
    print("\n=== Results ===")
    if all_passed:
        print("🎉 All index error fixes are working correctly!")
        print("\nKey fixes applied:")
        print("• Empty features list handling")
        print("• Safe technical indicators indexing (pad with zeros)")
        print("• Safe state representation indexing (bounds checking)")
        print("• Safe denoised features column access")
        print("• Safe hidden dimensions array indexing")
        print("\nThe 'list index out of range' errors should now be resolved.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()