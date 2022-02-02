import plotly as plt
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("/Users/Lillie/Desktop/CL Astro/ASTRO_INCLASS.csv")

print(df)


# Velocity Time Graph
fig = px.scatter(df, x = df["Day #"], y = df["Velocity (AU/day)"])
#fig.show()

#Distance Time Graph
fig = px.scatter(df, x = df["Day #"], y = df["Distance (AU)"])
#fig.show()

#3D plot of orbit

fig = px.scatter_3d(df, x='X', y='Y', z='Z', color = 'Velocity (AU/day)'
                    #color='', symbol='species'
                    )
dict = {"X": [0], 'Y': [0], 'Z':[0]}
df2 = pd.DataFrame(dict)

fig2 = px.scatter_3d(df2, x='X', y='Y', z='Z'
                    #color='', symbol='species'
                    )
fig3 = go.Figure(data=fig.data + fig2.data)

fig3.update_layout(
    title="Orbit of Jupiter about Sun",
    xaxis_title="X (AU)",
    yaxis_title="Y (AU)",
    #zaxis_title = 'Z (AU)',
    legend_title="Legend Title",
    #font=dict(
     #   family="Courier New, monospace",
      #  size=18,
     #   color="RebeccaPurple"
    #)
)


fig3.show()


