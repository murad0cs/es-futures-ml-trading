import numpy as np
import pandas as pd

# Simulate the issue
df_length = 283
signals_length = 4

# Create sample data
df = pd.DataFrame({'close': np.random.randn(df_length)}, 
                  index=range(df_length))
signals = pd.DataFrame({'signal': [1, 1, 1, 1]}, 
                       index=[10, 50, 100, 150])

# Simulate confidence scores for entire df
confidence_scores = np.random.rand(df_length)

print(f"DataFrame length: {len(df)}")
print(f"Signals length: {len(signals)}")
print(f"Confidence scores length: {len(confidence_scores)}")

# The old way that causes error:
try:
    filtered_signals_old = signals[confidence_scores >= 0.5]
    print("Old method worked (shouldn't happen)")
except ValueError as e:
    print(f"Old method error: {e}")

# The fixed way:
signal_confidence_scores = confidence_scores[signals.index]
filtered_signals_new = signals[signal_confidence_scores >= 0.5]
print(f"\nFixed method: Filtered {len(filtered_signals_new)} signals out of {len(signals)}")
print("Success! The fix works correctly.")