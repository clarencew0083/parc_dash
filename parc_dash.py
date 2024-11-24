import os
import plotly.graph_objects as go
import pymongo
import pandas as pd
import numpy as np
import json
import gridfs

class ParcDash:
    def __init__(self):
        self.MONGO_INITDB_ROOT_USERNAME = os.getenv('MONGO_INITDB_ROOT_USERNAME')
        self.MONGO_INITDB_ROOT_PASSWORD = os.getenv('MONGO_INITDB_ROOT_PASSWORD')

    def connect_to_mongo(self, from_scratch = False, host='localhost'):
        myclient = pymongo.MongoClient(f'mongodb://{self.MONGO_INITDB_ROOT_USERNAME}:{self.MONGO_INITDB_ROOT_PASSWORD}@{host}:27017/')
        mongo_parc = myclient['parc_dash']

        collist = mongo_parc.list_collection_names()
        if from_scratch and "sims" in collist:
            mongo_parc.parc_dash.drop()
        return mongo_parc
    
    def do_something(self):
                
        # assign directory
        directory = 'data/train/'
        
        # iterate over files in 
        # that directory
        for filename in os.scandir(directory):
            if filename.is_file():
                array = np.load(os.path.join(filename.path))

                # path = "data/train/void_100.npy"

                #data1 = np.load("../data/test/void_100.npy")

                # Load the NumPy array
                #array = np.load(os.path.join(path))  # Shape: (15, 5, 128, 256)

                # Initialize a list to store DataFrames for each channel
                channel_dataframes = []
                channel_list = ["temperature", "pressure", "microstructure", "velocity_x", "velocity_y"]

                # Iterate over each channel
                for channel_idx in range(array.shape[1]):  # 5 channels
                    # Extract data for the current channel
                    
                    channel_data = array[:, channel_idx, ...]  # Shape: (15, 128, 256)
                    # Create the timestep, x, y coordinates
                    timesteps, x_coords, y_coords = np.meshgrid(
                        np.arange(channel_data.shape[0]),       # 15 timesteps
                        np.arange(channel_data.shape[1]),       # 128 x-coordinates
                        np.arange(channel_data.shape[2]),       # 256 y-coordinates
                        indexing="ij"
                    )
                    
                    # Flatten all arrays
                    flattened_timestep = timesteps.flatten()
                    flattened_x = x_coords.flatten()
                    flattened_y = y_coords.flatten()
                    flattened_values = channel_data.flatten()
                    
                    # Create the DataFrame
                    df = pd.DataFrame({
                        "timestep": flattened_timestep,
                        "x": flattened_x,
                        "y": flattened_y,
                        "value": flattened_values
                    })
                    
                    # Add this channel's DataFrame to the list
                    channel_dataframes.append(df)
                    # upload channel data to mongo
                    self.upload_one_simulation_to_mongo(df, channel_list[channel_idx], filename)

                # Example: Accessing the DataFrame for the first channel
                first_channel_df = channel_dataframes[0]
                print(first_channel_df.head())

    def upload_one_simulation_to_mongo(self, df, channel_name, filename):
        mongo_parc = self.connect_to_mongo()
        print(mongo_parc.list_collection_names())
        result = df.to_json(orient="records")

        # Convert result from JSON string to Python object (list of records)
        result = json.loads(result)

        # Create a nested JSON structure
        new_json = {
            "variable": channel_name,
            "file": filename.name.split('.')[0],
            "Data": result  # Assign the parsed list of records here
        }

        large_json = json.dumps(new_json)

        # Use GridFS to handle large files
        fs = gridfs.GridFS(mongo_parc)

        # Store the JSON string in GridFS
        file_id = fs.put(large_json.encode(), filename=filename.name.split('.')[0])

        # Create a document with metadata to store in the collection
        metadata = {
            "variable": channel_name,
            "file": filename.name.split('.')[0],
            "file_id": file_id  # Store the GridFS file_id
        }


        # Insert metadata into the collection
        sims_collection = mongo_parc["sims"]  # Access the 'sims' collection
        sims_collection.insert_one(metadata)
        print(f"Uploaded {filename.name} to MongoDB")
        # Return the file_id and metadata inserted into the collection
        return metadata

    def query_simulation_from_mongo(self, channel_name, file_name):
        # Connect to MongoDB collection
        mongo_parc = self.connect_to_mongo()
        sims_collection = mongo_parc["sims"]  # Access the 'sims' collection

        # Query for documents where 'variable' matches channel_name and 'file' matches file_name
        query = {
            "variable": channel_name,
            "file": file_name
        }
        # Perform the query
        result = sims_collection.find(query)

        # Fetch all the matching documents
        documents = list(result)

        return documents

    def retrieve_file_from_mongo(self, file_id):
        mongo_parc = self.connect_to_mongo()
        fs = gridfs.GridFS(mongo_parc)

        # Retrieve the file using its file_id from GridFS
        file = fs.get(file_id)

        # Decode the content (JSON string) and load it as a Python object
        large_json = file.read().decode()
        data = json.loads(large_json)

        return data
    def display_ground_truth_plotly(ground_truth, batch_idx=0,):
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