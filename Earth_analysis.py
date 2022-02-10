import numpy as np
import pandas as pd
import plotly as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly_express as px
import plotly.graph_objects as go
from utilities import *
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.fft import fft

planet = 'earth'

ephemeris("/Users/Lillie/Desktop/CL Astro/Earth_Sun_Ephemeris.txt")

earth = vector_ephemeris_to_dataframe("/Users/Lillie/Desktop/CL Astro/Earth_Sun_Ephemeris.txt")

days = generate_days(earth)
distance = generate_distance(days)
velocity = generate_velocity(distance)
e = generate_eccentricity(velocity)

print("Eccentricity:", e)

#data = {"Days": days, "Distance (Au)": distance, "Velocity (Au/Day)": velocity}

df = pd.DataFrame(velocity)






# Velocity Time Graph

fig_v_t = px.scatter(df, x = df["days"], y = df["v"],
                     labels={
                         "days": "Time (Days)",
                         "v": "Velocity (AU/day)" },
                     title="Earth-Sun Distance Time Graph")

#fig_v_t.show()

#Distance Time Graph

fig_d_t = px.scatter(df, x = df["days"], y = df["d"],
                     labels={
                         "days": "Time (Days)",
                         "d": "Distance (AU)" },
                     title="Earth-Sun Distance Time Graph"
                     )
#fig_d_t.show()

#3D Plot

fig = px.scatter_3d(df, x='X', y='Y', z='Z', color = 'v' )
dict = {"X": [0], 'Y': [0], 'Z':[0]} #Sun at Center
df2 = pd.DataFrame(dict)
fig2 = px.scatter_3d(df2, x='X', y='Y', z='Z')
fig3 = go.Figure(data=fig.data + fig2.data)

#2d ellipse shape
fig = go.Figure()
fig.add_trace(go.Scatter(x = df['X'], y= df['Y'], mode = 'markers', name = 'Scatter Points'))
fig.add_trace(go.Trace(x=df['X'], y=df['Y'], name = 'Line of Best Fit'))
fig.add_trace(go.Scatter(x = df2['X'], y= df2['Y'], name = 'Sun', mode = 'markers+text', text = ['The Sun'], textposition= 'top center'))
#Find Equation of ellipse projection on X-Y plane


dat = find_elliptical_equation(df['X'], df['Y'])
fig.add_trace(go.Trace(x=dat['x_ranges'], y=dat['y_ranges'], name = 'Equation of Ellipse'))

fig.update_layout(title = f'Plot of X-Y plane of orbit of {planet}', showlegend=True)
#ellipse = go.Figure(data = fig_scatter.data + fig_line.data + fig_equation.data)




fig3.update_layout(
    title="Orbit of Earth about the Sun",
    xaxis_title="X (AU)",
    yaxis_title="Y (AU)",
    legend_title= "Velocity",
)


#fig3.show()

#Dash board

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="velocity_graph", figure=fig_v_t),
    dcc.Graph(id="distance_graph", figure=fig_d_t),
    dcc.Graph(id="3d_orbit", figure= fig3),
    dcc.Graph(id = 'elliptical_proof', figure = fig)

])

app.run_server(debug=True)