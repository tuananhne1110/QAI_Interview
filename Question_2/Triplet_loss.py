import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum((anchor - positive) ** 2)
    neg_dist = np.sum((anchor - negative) ** 2)
    loss = np.maximum(0, pos_dist - neg_dist + margin)
    return loss

def extended_triplet_loss(anchor, positives, negatives, margin=1.0):
    loss = 0
    for p in positives:
        for n in negatives:
            pos_dist = np.sum((anchor - p) ** 2)
            neg_dist = np.sum((anchor - n) ** 2)
            loss += np.maximum(0, pos_dist - neg_dist + margin)
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


# Test the implementation with 2 positives and 5 negatives
def test_extended_triplet_loss():
    anchor = np.array([1.0, 2.0, 3.0])
    positives = [np.array([1.1, 2.1, 3.1]), np.array([0.9, 1.9, 2.9])]
    negatives = [np.array([3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0]), 
                 np.array([1.5, 2.5, 3.5]), np.array([0.0, 1.0, 2.0]), 
                 np.array([4.0, 5.0, 6.0])]

    # Calculate the expected loss manually
    margin = 1.0
    expected_loss = 0

    # Manually compute the expected loss
    for p in positives:
        for n in negatives:
            pos_dist = np.sum((anchor - p) ** 2)
            neg_dist = np.sum((anchor - n) ** 2)
            loss_ij = np.maximum(0, pos_dist - neg_dist + margin)
            expected_loss += loss_ij

    # Calculate loss using the function
    calculated_loss = extended_triplet_loss(anchor, positives, negatives, margin)

    # Check if the calculated loss is as expected
    assert np.isclose(calculated_loss, expected_loss), f"Expected {expected_loss}, but got {calculated_loss}"
    print(f"Extended Triplet Loss Test Passed! Expected loss: {expected_loss}, Calculated loss: {calculated_loss}")


# Run the tests
test_triplet_loss()
test_extended_triplet_loss()
