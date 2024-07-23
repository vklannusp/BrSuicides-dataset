import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Color blindness transformation matrices
protanopia_matrix = np.array([
    [0.567, 0.433, 0],
    [0.558, 0.442, 0],
    [0, 0.242, 0.758]
])

deuteranopia_matrix = np.array([
    [0.625, 0.375, 0],
    [0.7, 0.3, 0],
    [0, 0.3, 0.7]
])

tritanopia_matrix = np.array([
    [0.95, 0.05, 0],
    [0, 0.433, 0.567],
    [0, 0.475, 0.525]
])

# Function to apply color blindness transformation
def simulate_colorblindness(color_rgb, transformation_matrix):
    # Apply transformation matrix
    transformed_rgb = np.dot(transformation_matrix, color_rgb)
    # Clip values to range [0, 1]
    transformed_rgb = np.clip(transformed_rgb, 0, 1)
    return transformed_rgb

# Function to generate a gradient palette
def generate_gradient_palette(start_hex, end_hex, n_colors=8):
    start_rgb = np.array([int(start_hex[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
    end_rgb = np.array([int(end_hex[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
    palette = [start_rgb * (1 - t) + end_rgb * t for t in np.linspace(0, 1, n_colors)]
    return palette

# Function to generate and display color palettes with colorblind simulation
def display_colorblind_palettes(start_hex, end_hex):
    # Monochromatic palettes
    monochromatic_palette_start = sns.light_palette(start_hex, n_colors=8, reverse=False)
    monochromatic_palette_end = sns.light_palette(end_hex, n_colors=8, reverse=False)

    # Gradient palette
    gradient_palette = generate_gradient_palette(start_hex, end_hex, n_colors=8)

    # Color names and transformation matrices
    color_names = ['Original', 'Protanopia', 'Deuteranopia', 'Tritanopia']
    transformation_matrices = [None, protanopia_matrix, deuteranopia_matrix, tritanopia_matrix]

    fig, axes = plt.subplots(6, 1, figsize=(6, 6))

    # Display monochromatic palettes
    axes[0].imshow([monochromatic_palette_start], aspect='auto')
    axes[0].set_title('Monochromatic Blue Palette')
    axes[0].axis('off')

    axes[1].imshow([monochromatic_palette_end], aspect='auto')
    axes[1].set_title('Monochromatic Orange Palette')
    axes[1].axis('off')

    # Display gradient palette
    axes[2].imshow([gradient_palette], aspect='auto')
    axes[2].set_title('Original Gradient Palette')
    axes[2].axis('off')

    # Display gradient palettes with colorblind simulation
    for ax, name, matrix in zip(axes[3:], color_names[1:], transformation_matrices[1:]):
        transformed_palette = [simulate_colorblindness(color, matrix) for color in gradient_palette]
        ax.imshow([transformed_palette], aspect='auto')
        ax.set_title(f'{name} Gradient Palette')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# COLOR SETTING

# Starting colors
my_blue = "#0011DC"
my_orange = "#FF7D00"

if __name__ == "__main__":
    # Display original and simulated colorblind palettes
    display_colorblind_palettes(my_blue, my_orange)