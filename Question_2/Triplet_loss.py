import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum((anchor - positive) ** 2)
    neg_dist = np.sum((anchor - negative) ** 2)
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return loss

# Test triplet_loss function
def test_triplet_loss():
    anchor = np.array([1.0, 2.0, 3.0])
    positive = np.array([1.1, 2.1, 3.1])
    negative = np.array([3.0, 4.0, 5.0])

    expected_loss = 0

    # Calculate loss using the function
    calculated_loss = triplet_loss(anchor, positive, negative)

    # Check if the calculated loss is as expected
    assert np.isclose(calculated_loss, expected_loss), f"Expected {expected_loss}, but got {calculated_loss}"
    print(f"Triplet Loss Test Passed! Expected loss: {expected_loss}, Calculated loss: {calculated_loss}")


# Run the tests
test_triplet_loss()

