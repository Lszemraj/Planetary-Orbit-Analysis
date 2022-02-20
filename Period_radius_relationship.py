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

# Working on Kepler's 3rd Law Relation

# Generate Dataframe of all periods and semimajor axis
...
list_of_file_names = ["Earth_Sun_Ephemeris.txt", "jupiter_final2.txt", "mars_final2.txt", "mercury_final.txt",
                      "neptune_final.txt", "saturn_final2.txt", "uranus_final.txt", "venus_final.txt"]

periods = []
p = {}
semis = []
ln_periods = []
ln_semis = []
ecc = {}
se = {}

for i in list_of_file_names:
    planet_data = load_data(i)
    days = generate_days(planet_data)
    distance = generate_distance(days)
    amp, omega, phase, offset, freq, period, fitfunc = fit_sin(distance['days'], distance['d'])
    semi = generate_semimajor_axis(distance)
    e = generate_eccentricity(distance)
    T = period
    periods.append(T)
    semis.append(semi)
    ecc[f'{i}'] = e
    p[f'{i}'] = T
    se[f'{i}'] = semi
    ln_periods.append(np.log(T))
    ln_semis.append(np.log(semi))

print("eccentricity", ecc)
print("periods", p)
print("semis", se)
data = {"Periods of Planets": periods, "Semimajor Axis of Planets": semis, "ln_t": ln_periods, "ln_s": ln_semis}
df = pd.DataFrame(data)

slope = ( df['ln_s'][1] - df['ln_s'][0] ) / (df['ln_t'][1] - df['ln_t'][0] )
#print("Slope, change in semi major axis over t", slope)
...

# Plotting
...
fig = go.Figure()
fig2 = go.Figure()


fig.add_trace(go.Scatter(x = (df['Periods of Planets'])**2, y= (df['Semimajor Axis of Planets'])**3,
                         mode = 'markers', name = 'Scatter Points'))
fig.add_trace(go.Trace(x= (df['Periods of Planets'])**2, y= (df['Semimajor Axis of Planets'])**3,
              name = 'Line of Best Fit'))
fig2.add_trace(go.Trace(x= np.log(df['Periods of Planets']), y= np.log(df['Semimajor Axis of Planets']),
              name = 'Slope = 2/3'))

fig.update_layout(title = "Period (T) Squared verses Semimajor Axis Cubed",
                  xaxis_title = 'T^2',
                  yaxis_title = '(Semimajor Axis)^3',
                  autosize=False,
                  width=1200, height=800
                  )
fig2.update_layout(title = "log Period (T)  verses log Semimajor Axis, slope demonstrates relationship to linearize",
                   xaxis_title = 'log(T)',
                   yaxis_title = 'log(semimajor axis)',
                   showlegend = True,
                   autosize=False,
                   width=1200, height=800
                   )
...

# Dash App
...
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="ln_relation", figure=fig2),
    dcc.Graph(id="power_relation", figure=fig)

])

app.run_server(debug=True)
...