from dash import Dash, html, dcc, Input, Output, Patch, clientside_callback, callback
import plotly.express as px
import plotly.io as pio
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from parc_dash import ParcDash
import pandas as pd

parc= ParcDash()

mongo_parc = parc.connect_to_mongo()
sims = mongo_parc["sims"]

display_variables  = sims.distinct( "variable" )
display_files  = sims.distinct( "file" )

# adds  templates to plotly.io
load_figure_template(["sketchy", "cyborg", "minty", "all", "minty_dark"])

df = px.data.gapminder()

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.FONT_AWESOME])

color_mode_switch =  html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch( id="color-mode-switch", value=False, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    ]
)

fig = px.scatter(
        df.query("year==2007"),
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        log_x=True,
        size_max=60,
        template="minty",
    )
controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Simulation Selection"),
                dbc.Select(
                    id="simulation_ddl",
                    options=display_files,
                    value="void_100",
                    className="dbc",
                ),
                dbc.Label("Variable Selection"),
                dbc.Select(
                    id="variable_ddl",
                    options=display_variables,
                    value="temperature",
                    className="dbc",
                ),   
            ],
        ),
    ],
    body=True,
)


app.layout = dbc.Container(
    [
        html.H1(["PARCv2 Dash"], className="bg-primary text-white h3 p-2"),
        color_mode_switch,
        html.Hr(),
         dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="simulation_graph",className="border"), md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@callback(
    Output("simulation_graph", "figure",  allow_duplicate=True),
    Input("color-mode-switch", "value"),
    prevent_initial_call=True
)
def update_figure_template(switch_on):
    # When using Patch() to update the figure template, you must use the figure template dict
    # from plotly.io  and not just the template name
    template = pio.templates["minty"] if switch_on else pio.templates["minty_dark"]

    patched_figure = Patch()
    patched_figure["layout"]["template"] = template
    return patched_figure

@app.callback(
    [Output(component_id='simulation_graph', component_property='figure')],
    [Input(component_id='simulation_ddl', component_property='value'),
     Input(component_id='variable_ddl', component_property='value')])
def plot_simulation(simulation, variable):
    documents = parc.query_simulation_from_mongo(variable, simulation)
    file_id = documents[0]["file_id"]
    data = parc.retrieve_file_from_mongo(file_id)
    df3 = data["Data"]
    df4 = pd.DataFrame(df3)
    return [parc.display_ground_truth_imshow(df4)]


clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');  
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)


if __name__ == "__main__":
    app.run_server(debug=True)