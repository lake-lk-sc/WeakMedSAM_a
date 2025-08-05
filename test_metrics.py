import torch
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the dice function from utils.metrics
from utils.metrics import dice

def test_dice_perfect_match():
    """Test dice coefficient with perfect match between result and reference."""
    # Create tensors with perfect match
    result = torch.zeros((2, 1, 4, 4))
    result[0, 0, 1:3, 1:3] = 1.0  # Create a 2x2 square in the middle of the first image
    result[1, 0, 0:2, 0:2] = 1.0  # Create a 2x2 square in the top-left of the second image
    
    reference = result.clone()  # Perfect match
    
    # Calculate dice coefficient
    dice_score, count = dice(result, reference)
    
    # Check if dice coefficient is 1.0 (perfect match)
    assert abs(dice_score / count - 1.0) < 1e-6, f"Expected dice score to be 1.0, but got {dice_score / count}"
    print("✓ test_dice_perfect_match passed")

def test_dice_no_match():
    """Test dice coefficient with no match between result and reference."""
    # Create tensors with no match
    result = torch.zeros((2, 1, 4, 4))
    result[0, 0, 0:2, 0:2] = 1.0  # Create a 2x2 square in the top-left of the first image
    result[1, 0, 0:2, 0:2] = 1.0  # Create a 2x2 square in the top-left of the second image
    
    reference = torch.zeros((2, 1, 4, 4))
    reference[0, 0, 2:4, 2:4] = 1.0  # Create a 2x2 square in the bottom-right of the first image
    reference[1, 0, 2:4, 2:4] = 1.0  # Create a 2x2 square in the bottom-right of the second image
    
    # Calculate dice coefficient
    dice_score, count = dice(result, reference)
    
    # Check if dice coefficient is 0.0 (no match)
    assert abs(dice_score / count) < 1e-6, f"Expected dice score to be 0.0, but got {dice_score / count}"
    print("✓ test_dice_no_match passed")

def test_dice_partial_match():
    """Test dice coefficient with partial match between result and reference."""
    # Create tensors with partial match
    result = torch.zeros((2, 1, 4, 4))
    result[0, 0, 0:3, 0:3] = 1.0  # Create a 3x3 square in the top-left of the first image
    result[1, 0, 0:3, 0:3] = 1.0  # Create a 3x3 square in the top-left of the second image
    
    reference = torch.zeros((2, 1, 4, 4))
    reference[0, 0, 1:4, 1:4] = 1.0  # Create a 3x3 square in the bottom-right of the first image
    reference[1, 0, 1:4, 1:4] = 1.0  # Create a 3x3 square in the bottom-right of the second image
    
    # Calculate dice coefficient
    dice_score, count = dice(result, reference)
    
    # Expected dice coefficient: 2 * (intersection) / (sum of areas) = 2 * 4 / (9 + 9) = 8 / 18 = 4/9 ≈ 0.444
    expected_dice = 4/9
    assert abs(dice_score / count - expected_dice) < 1e-2, f"Expected dice score to be {expected_dice}, but got {dice_score / count}"
    print("✓ test_dice_partial_match passed")

if __name__ == "__main__":
    print("Running tests for dice coefficient...")
    test_dice_perfect_match()
    test_dice_no_match()
    test_dice_partial_match()
    print("All tests passed!")