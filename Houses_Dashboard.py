from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import Dash
import pandas as pd
import plotly.express as px
import numpy as np

df = pd.read_csv("Inmuebles_data_studio.csv",skipinitialspace = True)

#leng variable utilizada para guardar número total de registros en csv.
leng =len(df)
#Asigna orden de los registros aleatoriamente. Frac = fracción de datos que devuelve.
#Devuelve todos al asignar frac = 1
df = df.sample(frac = 1)

#Imprime primeros 5 registros, a modo de muestra, después de asignar posiciones aleatoriamente


# Establece límites por característica para los datos que alimentarán al modelo.
# Se realiza este paso por datos recopilados poco creíbles o que no son
# afines al propósito del modelo. Ejemplo 22,000 km2 de terreno.


lim_sup_terr = 10000
lim_sup_con =10000
lim_sup_rec = 15
lim_sup_ba = 15
lim_sup_es = 15
lim_sup_precio = 150
lim_inf_precio = .5
df.drop(df[(df.Terreno>lim_sup_terr) | (df.Construccion > lim_sup_con) | (df.Recas > lim_sup_rec) | (df.Banhos > lim_sup_ba) | (df.Estacionamientos > lim_sup_es) | (df.Precio > lim_sup_precio) | (df.Terreno<15) | (df.Construccion<15) | (df.Recas == 0) | (df.Banhos == 0)  | (df.Precio < lim_inf_precio)].index, inplace=True)


res = len(df)
excluidos = leng -res

print(excluidos)

frecuencia = 1

df = df.groupby(['Delegacion','Colonia']).filter(lambda x : len(x)>frecuencia)
#df.groupby(['Delegacion','Colonia'])['Precio'].agg(['count'])

delegaciones_array = []
del_col_array = []

delegaciones = np.sort(df['Delegacion'].unique())
for delegacion in delegaciones:

    dic_del = {'label': None, 'value': None, 'dict': {}}
    dic_del['label'] = delegacion
    dic_del['value'] = delegacion

    del_df = df[df['Delegacion'] == delegacion]

    colonias_array = del_df['Colonia'].unique()
    del_col_array = []
    for colonia in colonias_array:
        dic_col = {'label': None, 'value': None}
        dic_col['label'] = colonia
        dic_col['value'] = colonia

        del_col_array.append(dic_col)

    dic_del['dict'] = del_col_array

    delegaciones_array.append(dic_del)

app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
delegaciones_dropdown = dcc.Dropdown(id='delegaciones_id', value='Cuauhtémoc', options=delegaciones_array)
colonias_dropdown = dcc.Dropdown(id='colonias_id', value='Roma')
grafico_dropdown = dcc.Dropdown(id='grafico_id', value='Scatter Plot',
                                options=[{'label': 'Scatter Plot', 'value': 'Scatter Plot'},
                                         {'label': 'Box Plot', 'value': 'Box Plot'},
                                         {'label': 'Histogram Plot', 'value': 'Histogram Plot'}])
variable_dropdown = dcc.Dropdown(id='variable_id', value='Precio', options=[{'label': 'Precio', 'value': 'Precio'},
                                                                            {'label': 'Terreno', 'value': 'Terreno'},
                                                                            {'label': 'Construccion',
                                                                             'value': 'Construccion'},
                                                                            {'label': 'Construccion',
                                                                             'value': 'Construccion'},
                                                                            {'label': 'Recamáras', 'value': 'Recas'},
                                                                            {'label': 'Baños', 'value': 'Banhos'},
                                                                            {'label': 'Estacionamientos',
                                                                             'value': 'Estacionamientos'},
                                                                            {'label': 'Medios Baños',
                                                                             'value': 'Medios Banhos'},
                                                                            {'label': 'Antigüedad',
                                                                             'value': 'Antiguedad'}])
scatter_delegaciones = dcc.Graph(id='scatter-graph')
box_delegaciones = dcc.Graph(id='box-graph')
histo_colonia = dcc.Graph(id='histo-graph')

# app.layout = html.Div(children=[
#   html.H1(children='House Prices in CDMX Dashboard'),
#   dcc.RadioItems(['Alvaro Obregón', 'Miguel Hidago', ], 'Alvaro Obregón', id = 'delegaciones_id', inline = True),
#   scatter_delegaciones

# ])


app.layout = dbc.Container([
    dbc.Row([html.Label("Precios de Casas en CDMX, México")],
            style={"margin": "0px 0px 0px 0px", "height": "10px", "width": "100%", "padding": "0px 0px 0px 0px",
                   "color": "white", "background-color": "MediumSeaGreen", "size": "18"}),
    dbc.Row([
        dbc.Col([
            html.Label('Delegaciones'),
            delegaciones_dropdown], width={"size": 3}),
        dbc.Col([
            html.Label('Colonia'),
            colonias_dropdown], width={"size": 4, "offset": 0}),
        dbc.Col([
            html.Label('Variable'),
            variable_dropdown], width={"size": 2, "offset": 0}),
        dbc.Col([
            html.Label('Gráfico'),
            grafico_dropdown], width={"size": 2, "offset": 0}),
    ], justify="start"),
    dbc.Row([
        dbc.Col([scatter_delegaciones], width={"size": 8}),
        dbc.Col([histo_colonia], width={"size": 4, "offset": 0}),
    ], justify="start")
])


#   dbc.Row([
#        dbc.Col([colonias_dropdown], width = 4),
#        dbc.Col([grafico_dropdown], width = 4)
#    ])
# app.layout = dbc.Container([markdown_title,colonias_dropdown,grafico_dropdown,variable_dropdown ,scatter_delegaciones,histo_colonia ])
# , dbc.Col([variable_dropdown], width = 4)]), dbc.Row([ dbc.Col([ colonias_dropdown ], width = 4), dbc.Col([variable_dropdown], width = 4)]),dbc.Row([ dbc.Col([ scatter_delegaciones ], width = 4), dbc.Col([histo_colonia], width = 4)]) ])
@app.callback(
    Output(component_id='colonias_id', component_property='options'),
    Input(component_id='delegaciones_id', component_property='value')
)
def colonias_dropwdown(selected_delegacion):
    delegaciones_array = []
    del_col_array = []

    delegaciones = np.sort(df['Delegacion'].unique())
    for delegacion in delegaciones:
        dic_del = {'label': None, 'value': None, 'dict': {}}
        dic_del['label'] = delegacion
        dic_del['value'] = delegacion

        del_df = df[df['Delegacion'] == delegacion]

        colonias_array = np.sort(del_df['Colonia'].unique())
        del_col_array = []
        for colonia in colonias_array:
            dic_col = {'label': None, 'value': None}
            dic_col['label'] = colonia
            dic_col['value'] = colonia

            del_col_array.append(dic_col)

        dic_del['dict'] = del_col_array

        delegaciones_array.append(dic_del)

        if delegacion == selected_delegacion:
            return del_col_array


@app.callback(
    Output(component_id='scatter-graph', component_property='figure'),
    Input(component_id='delegaciones_id', component_property='value'),
    Input(component_id='grafico_id', component_property='value'),
    Input(component_id='variable_id', component_property='value')
)
def scatter_box_graph(selected_delegacion, selected_grafico, selected_variable):
    filtered_df = df[df['Delegacion'] == selected_delegacion]

    if selected_grafico == 'Scatter Plot':
        fig = px.scatter(filtered_df, x='Colonia', y=selected_variable,
                         title=f'House Prices in {selected_delegacion} by district')

    elif selected_grafico == 'Box Plot':
        fig = px.box(filtered_df, x='Colonia', y=selected_variable,
                     title=f'House Prices in {selected_delegacion} by district')
    else:
        fig = px.histogram(filtered_df, x=selected_variable, title=f' Colonia: {selected_delegacion}')

    return fig


@app.callback(
    Output(component_id='histo-graph', component_property='figure'),
    Input(component_id='colonias_id', component_property='value'),
    Input(component_id='grafico_id', component_property='value'),
    Input(component_id='variable_id', component_property='value')
)
def box_colonia_graph(selected_colonia, selected_grafico, selected_variable):
    filtered_df = df[df['Colonia'] == selected_colonia]

    if selected_grafico == 'Scatter Plot':
        fig = px.scatter(filtered_df, x='Colonia', y=selected_variable, title=f' Colonia: {selected_colonia}')

    elif selected_grafico == 'Box Plot':
        fig = px.box(filtered_df, x='Colonia', y=selected_variable, title=f' Colonia: {selected_colonia}')
    else:
        fig = px.histogram(filtered_df, x=selected_variable, title=f' Colonia: {selected_colonia}')

    return fig


# dbc.Row([ dbc.Col([ colonias_dropdown ], width = 4),  dbc.Col([variable_dropdown], width = 4)])

if __name__ == '__main__':
    app.run_server(port=8050)
