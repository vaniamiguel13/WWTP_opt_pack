import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('f_output.csv')

# Rename columns for clarity
df.columns = ['Total Cost', 'Quality Index']

# Identify the three solutions and their row indices
economist_index = df['Total Cost'].idxmin()
economist_choice = df.loc[economist_index]

environmentalist_index = df['Quality Index'].idxmin()
environmentalist_choice = df.loc[environmentalist_index]

# Normalize the data for balanced solution calculation
df_norm = (df - df.min()) / (df.max() - df.min())
df_norm['distance'] = np.sqrt(df_norm['Total Cost']**2 + df_norm['Quality Index']**2)
balanced_index = df_norm['distance'].idxmin()
balanced_choice = df.loc[balanced_index]

# Create the plot
plt.figure(figsize=(14, 10))
plt.scatter(df['Total Cost'], df['Quality Index'], color='blue', alpha=0.6, label='All Solutions')

# Define offsets with x calculated using the provided formula
x_offset_value = 2.789883e5 * 1e-3
economist_offset = (x_offset_value - 230, 0.662514-10)
environmentalist_offset = (x_offset_value - 400, 0.662501+20)
balanced_offset = (x_offset_value - 240, 0.662503+13)

# Highlight the three chosen solutions and add annotations
solutions = [
    (economist_choice, economist_index, 'red', 'Economic Choice', economist_offset),
    (environmentalist_choice, environmentalist_index, 'green', 'Environmentalist Choice', environmentalist_offset),
    (balanced_choice, balanced_index, 'purple', 'Balanced Choice', balanced_offset)
]

for solution, index, color, label, offset in solutions:
    x, y = solution['Total Cost'], solution['Quality Index']
    # Create annotation with specified center coordinates
    plt.scatter(x, y, color=color, s=100, label=label)
    plt.annotate(f"Cost: {x:.6f}\nQI: {y:.6f}",
                 xy=(x, y),  # Position the annotation at the data point
                 textcoords="offset points",  # Use offset points for text placement
                 xytext=offset,  # Apply the computed offset
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='grey', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.xlabel('Total Cost')
plt.ylabel('Quality Index')
plt.title('Pareto Front with Highlighted Solutions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# Adjust layout and display
plt.tight_layout()
plt.savefig('pareto_front_with_choices_and_values.png', dpi=300)
plt.show()

# Print the values and row indices of the chosen solutions
for solution, index, _, label, _ in solutions:
    print(f"\n{label} (Row {index}):")
    print(f"Total Cost: {solution['Total Cost']:.6f}")
    print(f"Quality Index: {solution['Quality Index']:.6f}")
