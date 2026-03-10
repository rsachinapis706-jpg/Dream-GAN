import plotly.graph_objects as go

fig = go.Figure()

# --- Defines ---
node_width = 0.2
node_height = 0.12

def add_box(fig, x, y, text, bg_color, border_color):
    # Shadow
    fig.add_shape(type="rect",
        x0=x-node_width/2 + 0.005, y0=y-node_height/2 - 0.005,
        x1=x+node_width/2 + 0.005, y1=y+node_height/2 - 0.005,
        fillcolor="rgba(0,0,0,0.1)", line=dict(width=0), layer="below"
    )
    # Box
    fig.add_shape(type="rect",
        x0=x-node_width/2, y0=y-node_height/2, x1=x+node_width/2, y1=y+node_height/2,
        fillcolor=bg_color, line=dict(color=border_color, width=2),
        layer="below", opacity=1
    )
    # Text
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False,
        font=dict(family="Arial, sans-serif", size=14, color="#2D3748"),
        align="center",
        yshift=0
    )
    return (x,y)

def add_arrow(fig, start_x, start_y, end_x, end_y, path=None):
    if path:
        fig.add_shape(type="path", path=path, line=dict(color="#A0AEC0", width=3), layer="below")
        # Add arrow head at end specifically
        fig.add_annotation(
            x=end_x, y=end_y,
            ax=min(end_x, start_x) if end_x > start_x else max(end_x, start_x),
            ay=min(end_y, start_y) if end_y > start_y else max(end_y, start_y),
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#A0AEC0", standoff=0
        )
    else:
        fig.add_annotation(
            x=end_x, y=end_y, ax=start_x, ay=start_y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="#A0AEC0",
            standoff=5, startstandoff=5
        )

# --- Layout Grid ---
# Rows: y=0.8 (Top), y=0.5 (Middle), y=0.2 (Bottom)
# Cols: x=0.1, 0.35, 0.60, 0.85

color_data = "#E6FFFA"
border_data = "#319795"

color_model = "#EBF8FF"
border_model = "#3182CE"

color_ctrl = "#FFF5F5"
border_ctrl = "#E53E3E"

# Nodes Top Row
x_real, y_real = add_box(fig, 0.1, 0.8, "<b>Real EEG</b><br>Sequences ($X_{1:T}$)", color_data, border_data)
x_enc, y_enc = add_box(fig, 0.35, 0.8, "<b>GRU Encoder</b><br>$\mathcal{E}$", color_model, border_model)
x_lat, y_lat = add_box(fig, 0.60, 0.8, "<b>Latent Space</b><br>($H_{1:T}$)", color_data, border_data)
x_dec, y_dec = add_box(fig, 0.85, 0.8, "<b>GRU Decoder</b><br>$\mathcal{D}$", color_model, border_model)
x_rec, y_rec = add_box(fig, 1.1, 0.8, "<b>Reconstructed</b><br>($\tilde{X}_{1:T}$)", color_data, border_data)

# Nodes Bottom Row
x_noise, y_noise = add_box(fig, 0.1, 0.2, "<b>Gaussian Noise</b><br>($Z_{1:T}$) & Label ($Y$)", color_data, border_data)
x_gen, y_gen = add_box(fig, 0.35, 0.2, "<b>AC Generator</b><br>$\mathcal{G}$", color_model, border_model)
x_sup, y_sup = add_box(fig, 0.60, 0.2, "<b>Temporal</b><br><b>Supervisor</b> $\mathcal{S}$", color_model, border_model)

# Active Controls (Middle and End)
x_disc, y_disc = add_box(fig, 0.475, 0.5, "<b>Discriminator $\mathcal{D}_x$ &</b><br><b>Aux Classifier $\mathcal{C}$</b>", color_ctrl, border_ctrl)

x_paa, y_paa = add_box(fig, 1.1, 0.5, "<b>PAA Spectral</b><br><b>Governor</b>", color_ctrl, border_ctrl)
x_fin, y_fin = add_box(fig, 1.1, 0.2, "<b>Final Synthesized</b><br><b>Sequences</b> ($\hat{X}_{1:T}$)", color_data, border_data)

# --- Arrows (Direct) ---
add_arrow(fig, x_real+node_width/2, y_real, x_enc-node_width/2, y_enc)
add_arrow(fig, x_enc+node_width/2, y_enc, x_lat-node_width/2, y_lat)
add_arrow(fig, x_lat+node_width/2, y_lat, x_dec-node_width/2, y_dec)
add_arrow(fig, x_dec+node_width/2, y_dec, x_rec-node_width/2, y_rec)

add_arrow(fig, x_noise+node_width/2, y_noise, x_gen-node_width/2, y_gen)
add_arrow(fig, x_gen+node_width/2, y_gen, x_sup-node_width/2, y_sup)

add_arrow(fig, x_rec, y_rec-node_height/2, x_paa, y_paa+node_height/2)
add_arrow(fig, x_paa, y_paa-node_height/2, x_fin, y_fin+node_height/2)

# --- Arrows (Bent/Orthogonal Paths for neatness) ---
# Supervisor to Latent Space (upwards)
path_sup_lat = f"M {x_sup} {y_sup+node_height/2} V {y_lat-node_height/2-0.02} L {x_lat} {y_lat-node_height/2-0.02}" 
add_arrow(fig, x_sup, y_sup+node_height/2, x_lat, y_lat-node_height/2, path=path_sup_lat)

# Latent to Discriminator (downwards right)
path_lat_disc = f"M {x_lat-0.05} {y_lat-node_height/2} V {y_disc+node_height/2+0.05} L {x_disc} {y_disc+node_height/2+0.05}"
add_arrow(fig, x_lat-0.05, y_lat-node_height/2, x_disc, y_disc+node_height/2, path=path_lat_disc)

# Generator to Discriminator (upwards right)
path_gen_disc = f"M {x_gen} {y_gen+node_height/2} V {y_disc-node_height/2-0.05} L {x_disc} {y_disc-node_height/2-0.05}"
add_arrow(fig, x_gen, y_gen+node_height/2, x_disc, y_disc-node_height/2, path=path_gen_disc)

# --- Background & Layout Settings ---
fig.update_xaxes(visible=False, range=[0, 1.25])
fig.update_yaxes(visible=False, range=[0.05, 0.95])

fig.update_layout(
    width=1400,
    height=750,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20)
)

fig.write_image("results/system_arch_premium.png", scale=3)
print("Saved premium architecture image at results/system_arch_premium.png")
