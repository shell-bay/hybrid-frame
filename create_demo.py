"""Create an animated GIF demonstrating HybridFrame capabilities."""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from IPython.display import HTML, display

# Create figure with dark theme
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('HybridFrame: DuckDB + Pandas', fontsize=20, fontweight='bold', color='#00d4ff')

# Sample data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'value': np.random.uniform(0, 100, n)
})

# Animation frames
frames = []

def animate(frame_num):
    plt.clf()
    
    if frame_num < 30:
        # Frame 1-30: Show raw data
        ax = plt.subplot(111)
        ax.scatter(data['x'][:100], data['y'][:100], c=data['value'][:100], 
                   cmap='viridis', alpha=0.6, s=50)
        ax.set_title('Raw Pandas DataFrame', fontsize=14, color='#00d4ff')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.text(0.02, 0.98, f'Shape: {data.shape}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, color='white')
        
    elif frame_num < 60:
        # Frame 31-60: Show DuckDB filter
        ax = plt.subplot(111)
        filtered = data[data['x'] > 0]
        ax.scatter(filtered['x'][:100], filtered['y'][:100], 
                   c=filtered['value'][:100], cmap='viridis', alpha=0.6, s=50)
        ax.set_title('DuckDB Filter: x > 0', fontsize=14, color='#00d4ff')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.text(0.02, 0.98, f'Filtered: {filtered.shape[0]} rows', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10, color='white')
        
    elif frame_num < 90:
        # Frame 61-90: Show grouped aggregation
        ax = plt.subplot(111)
        grouped = data.groupby('category')['value'].mean()
        bars = ax.bar(grouped.index, grouped.values, color=['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffe66d'])
        ax.set_title('DuckDB GroupBy Aggregation', fontsize=14, color='#00d4ff')
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Value')
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', color='white')
    
    else:
        # Frame 91-120: Show final result
        ax = plt.subplot(111)
        result = data[data['x'] > 0].groupby('category')['value'].mean()
        bars = ax.bar(result.index, result.values, color=['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffe66d'])
        ax.set_title('HybridFrame Result', fontsize=14, color='#00d4ff')
        ax.set_xlabel('Category')
        ax.set_ylabel('Mean Value (x > 0)')
        for bar, val in zip(bars, result.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', color='white')
        ax.text(0.02, 0.02, 'Ready for ML!', transform=ax.transAxes,
                verticalalignment='bottom', fontsize=12, color='#4ecdc4', fontweight='bold')
    
    plt.tight_layout()

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=120, interval=50, repeat=True)

# Save as GIF
ani.save('hybrid_frame_demo.gif', writer='pillow', fps=20, dpi=100)
plt.close()

print("GIF saved as hybrid_frame_demo.gif")
