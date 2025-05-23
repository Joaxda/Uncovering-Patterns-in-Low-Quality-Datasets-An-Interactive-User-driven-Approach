import dash
import dash_auth
from dash import html, dcc
import dash_bootstrap_components as dbc
import diskcache

from utils.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK, SECRET_KEY, VALID_USERNAME, VALID_PASSWORD
# Load credentials from environment variables (or use defaults)
VALID_USERNAME_PASSWORD_PAIRS = {
    VALID_USERNAME: VALID_PASSWORD
}

# You can also load additional settings from your utils/settings.py if needed.
# For example:
# from utils.settings import APP_HOST, APP_PORT, APP_DEBUG

# Use an external stylesheet (you can choose Bootstrap or any other)
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Create the Dash app with multi-page support.
cache = diskcache.Cache("/cache")

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    use_pages=True,  # This enables the Dash pages functionality.
    suppress_callback_exceptions=True,
)

# Apply BasicAuth to the entire app.
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
app.server.secret_key = SECRET_KEY
# Define the app layout.
app.layout = html.Div([
    # The dcc.Location component tracks the URL.
    dcc.Location(id='url', refresh=True),
    
    # Navigation bar for your multi-page app.
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Dataset", href="/")),
            dbc.NavItem(dbc.NavLink("Data processing", href="/processing")),
        ],
        brand="Data Analysis Dashboard",
        #brand_href="/",
        color="dark",
        dark=True,
        className="mb-2",
    ),
    
    # This container will be automatically populated by pages
    # stored in the `pages/` directory.
    html.Div(dash.page_container)
])

server = app.server  # Expose the Flask server (if deploying with gunicorn, etc.)

if __name__ == '__main__':
    # You can specify the host and port as needed or load them from settings.
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check= DEV_TOOLS_PROPS_CHECK
    )
