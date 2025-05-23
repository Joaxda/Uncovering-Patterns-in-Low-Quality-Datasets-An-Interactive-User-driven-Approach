from dash import dcc, html
from dash.dependencies import Input, Output
from app import app
from pages import dataset, processing

# Dont removed even if they look unused.
import callbacks.datasetCallbacks
import callbacks.processingCallbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/processing':
        return processing.layout
    else: 
        return dataset.layout

if __name__ == '__main__':
    app.run_server(debug=True)
