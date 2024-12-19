import threading
import time
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import curdoc

# Shared state variable for the next value (thread-safe)
next_value = [0]

# Create a ColumnDataSource with some initial data
source = ColumnDataSource(data=dict(values=[next_value[0]]))

# Create a Div to display dynamic content
div = Div(text=f"<h1>Current Value: {source.data['values'][0]}</h1>", width=400, height=100)

# Function to patch the data source
def update_data():
    global next_value
    current_value = source.data['values'][0]
    if next_value[0] > current_value:  # Only update if the thread has set a new value
        print(f"Patching data source with new value: {next_value[0]}")
        source.patch({'values': [(0, next_value[0])]})  # Patch the data source
        update_div()  # Update the Div content

# Function to update the Div content
def update_div():
    new_value = source.data['values'][0]
    div.text = f"<h1>Current Value: {new_value}</h1>"
    print(f"Updated Div with new value: {new_value}")  # Debugging print

# Thread function that changes the shared state
def background_thread():
    global next_value
    print("Background thread started...")
    while True:
        time.sleep(3)  # Simulate delay or external event
        new_value = next_value[0] + 1
        print(f"Background thread: New value is {new_value}")
        next_value[0] = new_value  # Update the shared state

# Start the background thread
def start_background_thread():
    print("Starting background thread...")
    thread = threading.Thread(target=background_thread)
    thread.daemon = True
    thread.start()

# Start the background thread
start_background_thread()

# Periodic callback to apply updates in the main thread
curdoc().add_periodic_callback(update_data, 500)  # Check for updates every 500ms

# Layout the components
layout = column(div)

# Add the layout to the current document (to be served by Bokeh server)
curdoc().add_root(layout)

# Debugging to see when the document is loaded
print("Bokeh server app started and periodic callback is running.")
