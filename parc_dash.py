import plotly.graph_objects as go

class ParcDash:
    def __init__(self):
        pass
    def display_ground_truth_plotly(
    ground_truth,
    #channels,
    #cmaps,
    batch_idx=0,
    ):
        channels = ["pressure", "Reynolds", "u", "v"]  # Adjust as per your data
        cmaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
        ]  # Adjust as per your preference
        ground_truth_sequence = ground_truth[:, batch_idx].cpu().numpy()  # (timesteps, channels, height, width)
        timesteps = len(ground_truth['timestep'].unique())
        num_channels, height, width = ground_truth_sequence.shape()
        print(f"timesteps: {timesteps}, num_channels: {num_channels}, height: {height}, width: {width}")

        for i, channel_name in enumerate(channels):
            cmap = cmaps[i]

            # Create a figure with frames
            fig = go.Figure()

            # Add the first frame for initialization
            gt_frame = ground_truth_sequence[0, i]
            fig.add_trace(
                go.Heatmap(
                    z=gt_frame,
                    colorscale=cmap,
                    showscale=True,
                    name="Ground Truth",
                )
            )

            # Add animation frames
            frames = []
            for t in range(timesteps):
                gt_frame = ground_truth_sequence[t, i]
                frames.append(
                    go.Frame(
                        data=[
                            go.Heatmap(z=gt_frame, colorscale=cmap, showscale=False),
                        ],
                        name=f"frame_{t}",
                    )
                )

            fig.frames = frames

            # Update layout
            fig.update_layout(
                title=f"Channel: {channel_name} - Ground Truth",
                xaxis=dict(title="Width", 
                        showgrid=False,
                        zeroline=False,
                        visible=False,
                        showticklabels=False,
                        range=[0, width]),
                yaxis=dict(title="Height", 
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        visible=False,
                        scaleanchor="x", 
                        range=[height, 0]),
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                                ],
                            },
                        ],
                    }
                ],
            )

            # Add slider for controlling frames
            sliders = [
                {
                    "steps": [
                        {
                            "args": [
                                [f"frame_{t}"],
                                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                            ],
                            "label": f"Timestep {t + 1}",
                            "method": "animate",
                        }
                        for t in range(timesteps)
                    ],
                    "active": 0,
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "yanchor": "top",
                    "y": -0.2,
                }
            ]

            fig.update_layout(sliders=sliders)

            # Show the figure
            fig.show()