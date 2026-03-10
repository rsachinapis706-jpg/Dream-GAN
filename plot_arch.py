import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup the figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 16)
ax.set_ylim(-1, 6)
ax.axis('off')

# Define styles
def draw_box(ax, x, y, width, height, text, bg_color, border_color, font_size=11, font_weight='bold'):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.1, y-0.1), width, height, boxstyle="round,pad=0.2",
                                    edgecolor='none', facecolor='gray', alpha=0.3)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2",
                                 edgecolor=border_color, facecolor=bg_color, linewidth=2)
    ax.add_patch(box)
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=font_size, fontweight=font_weight, color='#333333', wrap=True)
    return box

def draw_arrow(ax, start, end, connectionstyle="arc3"):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#555555", lw=2.5, connectionstyle=connectionstyle))

# Color Palettes
color_data_bg = "#E8F5E9"
color_data_border = "#4CAF50"

color_model_bg = "#E3F2FD"
color_model_border = "#2196F3"

color_control_bg = "#FCE4EC"
color_control_border = "#E91E63"

# Node coordinates and details (x, y, width, height)
nodes = {
    "real": {"pos": (1, 4), "size": (2.5, 1), "text": "Real EEG\nSequences ($X_{1:T}$)", "type": "data"},
    "encoder": {"pos": (4.5, 4), "size": (2, 1), "text": "GRU Encoder $\mathcal{E}$", "type": "model"},
    "latent": {"pos": (7.5, 4), "size": (2.5, 1), "text": "Latent Space ($H_{1:T}$)", "type": "data"},
    "decoder": {"pos": (11, 4), "size": (2, 1), "text": "GRU Decoder $\mathcal{D}$", "type": "model"},
    "recon": {"pos": (14, 4), "size": (2.5, 1), "text": "Reconstructed\n($\\tilde{X}_{1:T}$)", "type": "data"},
    
    "noise": {"pos": (1, 1), "size": (2.5, 1), "text": "Gaussian Noise ($Z_{1:T}$)\n& Label ($Y$)", "type": "data"},
    "generator": {"pos": (4.5, 1), "size": (2, 1), "text": "AC Generator $\mathcal{G}$", "type": "model"},
    "supervisor": {"pos": (7.5, 1), "size": (2.5, 1), "text": "Temporal\nSupervisor $\mathcal{S}$", "type": "model"},
    
    # Place Discriminator centrally between Latent and Generator
    "discriminator": {"pos": (5.5, 2.5), "size": (3, 0.8), "text": "Discriminator $\mathcal{D}_x$ &\nAux Classifier $\mathcal{C}$", "type": "control"},
    
    # Place PAA
    "paa": {"pos": (14, 2), "size": (2.5, 1), "text": "PAA Spectral\nGovernor", "type": "control"},
    "final": {"pos": (14, 0), "size": (2.5, 1), "text": "Final Synthesized\nSequences ($\hat{X}_{1:T}$)", "type": "data"}
}

# Draw Nodes
for key, node in nodes.items():
    x, y = node["pos"]
    w, h = node["size"]
    
    if node["type"] == "data":
        bg, border = color_data_bg, color_data_border
    elif node["type"] == "model":
        bg, border = color_model_bg, color_model_border
    else:
        bg, border = color_control_bg, color_control_border
        
    draw_box(ax, x, y, w, h, node["text"], bg, border)

# Define Arrows (Start Center to End Center logic)
def get_center(node):
    return (node["pos"][0] + node["size"][0]/2, node["pos"][1] + node["size"][1]/2)

c_real = get_center(nodes["real"])
c_encoder = get_center(nodes["encoder"])
c_latent = get_center(nodes["latent"])
c_decoder = get_center(nodes["decoder"])
c_recon = get_center(nodes["recon"])

c_noise = get_center(nodes["noise"])
c_generator = get_center(nodes["generator"])
c_supervisor = get_center(nodes["supervisor"])

c_disc = get_center(nodes["discriminator"])
c_paa = get_center(nodes["paa"])
c_final = get_center(nodes["final"])

# Draw specific paths
# Top Row
draw_arrow(ax, (c_real[0]+1.25, c_real[1]), (c_encoder[0]-1.0, c_encoder[1]))
draw_arrow(ax, (c_encoder[0]+1.0, c_encoder[1]), (c_latent[0]-1.25, c_latent[1]))
draw_arrow(ax, (c_latent[0]+1.25, c_latent[1]), (c_decoder[0]-1.0, c_decoder[1]))
draw_arrow(ax, (c_decoder[0]+1.0, c_decoder[1]), (c_recon[0]-1.25, c_recon[1]))

# Bottom Row
draw_arrow(ax, (c_noise[0]+1.25, c_noise[1]), (c_generator[0]-1.0, c_generator[1]))
draw_arrow(ax, (c_generator[0]+1.0, c_generator[1]), (c_supervisor[0]-1.25, c_supervisor[1]))

# Generator/Supervisor to Latent
draw_arrow(ax, (c_supervisor[0], c_supervisor[1]+0.5), (c_latent[0], c_latent[1]-0.5))

# Latent -> Disc
draw_arrow(ax, (c_latent[0]-0.2, c_latent[1]-0.5), (c_disc[0]+1.0, c_disc[1]+0.2), connectionstyle="angle,angleA=-90,angleB=0,rad=10")

# Generator -> Disc
draw_arrow(ax, (c_generator[0]+0.2, c_generator[1]+0.5), (c_disc[0]-1.0, c_disc[1]-0.2), connectionstyle="angle,angleA=90,angleB=180,rad=10")

# Recon -> PAA -> Final
draw_arrow(ax, (c_recon[0], c_recon[1]-0.5), (c_paa[0], c_paa[1]+0.5))
draw_arrow(ax, (c_paa[0], c_paa[1]-0.5), (c_final[0], c_final[1]+0.5))

# Legend
leg_x, leg_y = 1, -0.5
draw_box(ax, leg_x, leg_y, 1.5, 0.4, "Data/Signals", color_data_bg, color_data_border, font_size=9)
draw_box(ax, leg_x+2, leg_y, 1.5, 0.4, "Core Models", color_model_bg, color_model_border, font_size=9)
draw_box(ax, leg_x+4, leg_y, 1.5, 0.4, "Supervisors/Constraints", color_control_bg, color_control_border, font_size=9)

# Save
plt.tight_layout()
plt.savefig('results/system_arch.png', dpi=300, bbox_inches='tight')
print("Successfully saved manual architecture image to results/system_arch.png")
