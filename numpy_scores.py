# numpy_scores.py
import numpy as np

# ------------------------
# Task 1 — Generate & Inspect
# ------------------------
np.random.seed(42)

# random integers between 50 and 100 inclusive -> use high=101 (exclusive)
scores = np.random.randint(50, 101, size=(5, 4))

print("Scores:\n", scores)

# The score of the 3rd student in the 2nd subject (0-based indexing -> [2, 1])
print("\nTask 1.1: 3rd student, 2nd subject =", scores[2, 1])

# All scores of the last 2 students (rows 3 and 4)
print("\nTask 1.2: Last 2 students (all subjects):\n", scores[-2:, :])

# All scores for the first 3 students in subjects 2 and 3 only (cols 1 and 2)
print("\nTask 1.3: First 3 students, subjects 2 & 3:\n", scores[:3, 1:3])

# ------------------------
# Task 2 — Broadcasting Analysis
# ------------------------
# Column-wise mean (average per subject), rounded to 2 decimals
col_means = np.round(scores.mean(axis=0), 2)
print("\nTask 2.1: Column-wise means =", col_means)

# Add curve using broadcasting, cap at 100
curve = np.array([5, 3, 7, 2])
curved_scores = np.clip(scores + curve, None, 100)
print("\nTask 2.2: Curved scores (capped at 100):\n", curved_scores)

# Row-wise max (best subject per student)
row_max = curved_scores.max(axis=1)
print("\nTask 2.3: Row-wise max =", row_max)

# ------------------------
# Task 3 — Normalize & Identify
# ------------------------
# Min-max normalization per row: (x - row_min) / (row_max - row_min)
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)
normalized = (curved_scores - row_min) / (row_max - row_min)

print("\nTask 3.1: Normalized (per row) scores:\n", normalized)

# Identify student index (row) and subject index (col) of the single highest value
flat_argmax = normalized.argmax()  # first max in row-major order
student_idx, subject_idx = np.unravel_index(flat_argmax, normalized.shape)
print("\nTask 3.2: Highest normalized value at (student, subject) =",
      (student_idx, subject_idx),
      "value =", normalized[student_idx, subject_idx])

# Extract all curved_scores strictly above 90 as 1D array
above_90 = curved_scores[curved_scores > 90]
print("\nTask 3.3: curved_scores > 90 =", above_90)
