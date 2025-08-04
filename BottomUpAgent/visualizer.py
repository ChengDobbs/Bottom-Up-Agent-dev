import yaml
import json
from PIL import Image
import io
import base64
from flask import Flask, request
from dash import Dash, html, dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import requests
import pyautogui
import datetime
import os
from BottomUpAgent.Eye import Eye
from dash import ctx

# ---- Load config.yaml ----
with open('config/sts_explore_claude.yaml', 'r') as f:
    default_config = yaml.safe_load(f)

# ---- Initialize Eye ----
eye = Eye(default_config)

# ---- Global State ----
global_data = {
    'config': default_config,
    'step': '',
    'potential_actions': '',        
    'temperature': '',     
    'decision': '',       
    'action_goal': {},
    'suspend_actions': [],
    'candidate_actions': [],
    'selected_action_id': None,
    'result_tree': {},
    'delete_ids': [],
    'exec_chain': [],           # merged: screen, operation
    'explore_tree': [],         # merged: name, state, children
    'result': ''                # new: to store the result of operations
}

RECORD_DIR = os.path.join(os.getcwd(), default_config['result_path'], 
                          default_config['game_name'], default_config['run_name'])
recording = False

# Playback states
playback_mode = False
playback_folder = ''
playback_files = []
playback_index = 0


# ---- Flask + Dash Setup ----
server = Flask(__name__)
app = Dash(__name__, server=server)

# ---- API Endpoint for Updates ----
@server.route('/api/update', methods=['POST'])
def update_data():
    data = request.get_json()
    for key, value in data.items():
        if key in global_data:
            global_data[key] = value
    return {'status': 'ok'}


# ---- Screenshot Utility ----
def capture_frame():
    """Capture a frame from the game window."""
    try:
        # Try to find the game window using Eye's cross-platform method
        window_info = None

        # Use the window name from eye instance (configured from config)
        if eye.window_name:
            window_info = eye.find_window_cross_platform(eye.window_name)

        if window_info:
            # Use the found window information
            left = window_info['left']
            top = window_info['top']
            width = window_info['width']
            height = window_info['height']

            # Capture the window using pyautogui (consistent with Eye.py)
            # pyautogui.screenshot() returns PIL Image, which is already in RGB format
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            return screenshot
        else:
            # Fallback to full screen capture
            screenshot = pyautogui.screenshot()
            return screenshot
    except Exception as e:
        print(f"Error capturing frame: {e}")
        # Fallback to full screen capture
        screenshot = pyautogui.screenshot()
        return screenshot


def encode_image(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()
    return 'data:image/png;base64,' + encoded

# ---- Tree to Cytoscape Elements ----
def tree_to_cyto(data, parent=None, elements=None):
    if elements is None:
        elements = []
    node_id = str(id(data))
    state_colors = {
        'Potential': '#f0f0f0',
        'Selected': '#add8e6',
        'New': '#ccffcc',
        'Fail': '#ffcccc'
    }
    bg = state_colors.get(data.get('state', ''), '#ffffff')
    elements.append({
        'data': {'id': node_id, 'label': f"{data.get('name','')}"},
        'style': {
            'background-color': bg,
            'width': '150px',
            'height': '80px',
            'text-wrap': 'wrap',        
            'white-space': 'pre-wrap',   
            'text-max-width': '120px' 
        }
    })
    if parent:
        elements.append({'data': {'source': parent, 'target': node_id}})
    for child in data.get('children', []):
        tree_to_cyto(child, parent=node_id, elements=elements)
    return elements

# ---- Dash Layout ----
app.layout = html.Div(style={'font-family': 'Arial, sans-serif', 'margin': '20px'}, children=[
    dcc.Interval(id='interval', interval=2000, n_intervals=0),
    html.H1("Multimodal Agent Visualizer"),

    html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
        html.Button('▶️ Start Record', id='start-record'),
        html.Button('⏹️ Stop Record', id='stop-record'),
        html.Button('🎥 Start Play', id='start-playback'),
        html.Button('⏸️ Stop Play', id='stop-playback'),
        dcc.Input(id='playback-folder', type='text', placeholder='Input folder name', style={'width': '200px'}),
        html.Div(id='record-status', style={'align-self': 'center', 'font-weight': 'bold'})
    ]),

    # Top row: Interface, Run States & Actions, Explore Tree
    html.Div(style={'display': 'flex'}, children=[
        # Game Interface
        html.Div(style={'flex': 1, 'padding-right': '20px'}, children=[
            html.H2("🎮 Game Interface"),
            html.Img(id='game-image', style={'width': '800px', 'border': '1px solid #ccc'}),
        ]),

        # Run States, Action Goal
        html.Div(style={'flex': 1, 'padding': '0 20px'}, children=[
            html.H2("🏃 Run States"),
            html.Div(id='run-states', style={'display': 'flex', 'gap': '20px'}),
            html.H2("🎯 Action Goal"),
            html.Div(id='action-goal'),

        ]),

        # Explore Tree
        html.Div(style={'flex': 1, 'padding-left': '20px'}, children=[
            html.H2("🌳 Explore Tree"),
            cyto.Cytoscape(
                id='explore-tree', layout={'name': 'breadthfirst', 'directed': True, 'padding': 10},
                style={'width': '100%', 'height': '400px'}, elements=[],
                stylesheet=[
                    {'selector': 'node', 'style': {'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center'}},
                    {'selector': 'edge', 'style': {'target-arrow-shape': 'triangle', 'curve-style': 'bezier'}}
                ]
            )
        ])
    ]),

    # second row: Candidate Actions & Exec Chain + LLM + Results
    html.Div(style={'display': 'flex'}, children=[
        # Candidate Actions 
        html.Div(style={'flex': 1}, children=[
            html.H2("📋 Candidate Actions"),
            html.Div(id='candidates', style={'display': 'flex', 'flex-wrap': 'wrap'})
        ]),

        #Exec Chain + LLM + Results
        html.Div(style={'flex': 1}, children=[
            html.H2("➡️ Exec Chain & LLM"),
            html.Div(id='exec-chain', style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'center'}),
            html.H2("📝 Result"),
            html.Pre(id='result-text', style={'whiteSpace': 'pre-wrap'})
        ]),
    ]),

    # third row: config
    html.Div(
        children=[
            html.H2("⚙️ Config"),
            html.Pre(id='config', style={'whiteSpace': 'pre-wrap'})
        ]
    )
])

# ---- Callback to Refresh UI ----
@app.callback(
    Output('game-image', 'src'),
    Output('config', 'children'),
    Output('action-goal', 'children'),
    Output('run-states', 'children'),
    Output('candidates', 'children'),
    Output('explore-tree', 'elements'),
    Output('exec-chain', 'children'),
    Output('result-text', 'children'),
    Input('interval', 'n_intervals')
)
def update_ui(n):
    global recording, playback_mode, playback_folder, playback_files, playback_index

    use_record = None

    # Reply mode
    if playback_mode and playback_folder:
        print(playback_folder)
        folder_path = os.path.join(RECORD_DIR, playback_folder)
        if not playback_files:
            if os.path.exists(folder_path):
                playback_files = sorted(os.listdir(folder_path))
                playback_index = 0

        print(len(playback_files)) 
        if playback_index < len(playback_files):
            filename = playback_files[playback_index]
            path = os.path.join(folder_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            use_record = record

            # update global_data using record
            for key in ['config', 'step', 'potential_actions', 'temperature', 'decision', 'action_goal', 'suspend_actions', 'candidate_actions', 'result_tree', 'delete_ids', 'exec_chain', 'explore_tree', 'result']:
                if key in record:
                    global_data[key] = record[key]
            if 'LLM' in record:
                global_data['LLM'] = record['LLM']

            playback_index += 1

    # use_record to update global_data

    # 1. Pic
    if use_record and 'screenshot' in use_record:
        screenshot = use_record['screenshot']
    else:
        img = capture_frame()
        screenshot = encode_image(img)

    # 2. Config
    config_str = json.dumps(global_data['config'], indent=2, ensure_ascii=False)

    # 3. Action Goal
    ag = global_data['action_goal']
    ag_div = [html.P(f"ID: {ag.get('id', '')}"), html.P(f"Name: {ag.get('name', '')}"), html.P(f"Description: {ag.get('description', '')}")]

    # 4. Run States
    run_divs = [
        html.Div([html.H4("Step"), html.P(str(global_data.get('step', '')))]),
        html.Div([html.H4("Potential Actions"), html.P(str(global_data.get('potential_actions', '')))]),
        html.Div([html.H4("Temperature"), html.P(str(global_data.get('temperature', '')))]),
        html.Div([html.H4("Decision"), html.P(str(global_data.get('decision', '')))])
    ]
    run_states_div = html.Div(run_divs, style={'display': 'flex', 'gap': '10px'})

    # 5. Candidate Actions
    selected_action_id = global_data.get('selected_action_id', None)
    cand = []
    for a in global_data['candidate_actions']:
        bg = '#add8e6' if a['id'] == selected_action_id else '#f0f0f0'
        if a['id'] in global_data['suspend_actions']:
            bg = '#ccffcc'
        if a['id'] in global_data['delete_ids']:
            bg = '#ffcccb'
        cand.append(html.Div([
            html.P(a['name'], style={'font-weight': 'bold'}),
            html.P(f"ID: {a['id']}"),
            html.P(f"fitness: {a.get('fitness', '')}"),
            html.P(f"num: {a.get('num', '')}"),
            html.P(f"prob: {a.get('prob', '')}"),
            ], style={'backgroundColor': bg, 'padding': '5px', 'margin': '2px'}))

    # 6. Explore Tree
    explore_data = global_data.get('explore_tree')
    explore_elements = tree_to_cyto(explore_data) if explore_data else []

    # 7. Exec Chain
    exec_steps = global_data.get('exec_chain', [])
    exec_div = []
    for step in exec_steps:
        if 'screen' in step:
            exec_div.append(html.Img(src=step['screen'], style={'width': '200px'}))
        if 'operation' in step and step['operation'] is not None:
            exec_div.append(html.Pre(json.dumps(step['operation'], indent=2)))

    # 8. Result Text
    result_txt = json.dumps(global_data.get('result', ''), indent=2, ensure_ascii=False)

    # 9. Save conditionally
    if recording and not playback_mode: 
        record = {
            'screenshot': screenshot,
            'config': global_data['config'],
            'action_goal': global_data['action_goal'],
            'step': global_data['step'],
            'potential_actions': global_data['potential_actions'],        
            'temperature': global_data['temperature'],     
            'decision': global_data['decision'],      
            'candidate_actions': global_data['candidate_actions'],
            'explore_tree': global_data['explore_tree'],
            'exec_chain': global_data['exec_chain'],
            'LLM': global_data.get('LLM', []),
            'result': global_data.get('result', {})
        }
        record_data(record)

    return screenshot, config_str, ag_div, run_states_div, cand, explore_elements, exec_div, result_txt

@app.callback(
    Output('record-status', 'children', allow_duplicate=True),
    Input('start-record', 'n_clicks'),
    Input('stop-record', 'n_clicks'),
    Input('start-playback', 'n_clicks'),
    Input('stop-playback', 'n_clicks'),
    Input('playback-folder', 'value'),
    prevent_initial_call=True
)
def control_buttons(start_rec, stop_rec, start_play, stop_play, folder_name):
    global recording, playback_mode, playback_folder, playback_files, playback_index

    triggered = ctx.triggered_id

    if triggered == 'start-record':
        recording = True
    elif triggered == 'stop-record':
        recording = False
    elif triggered == 'start-playback':
        if folder_name:
            playback_folder = folder_name
            playback_files = []
            playback_index = 0
            playback_mode = True
    elif triggered == 'stop-playback':
        playback_mode = False

    return ("Recording" if recording else "Not Recording") + " | " + ("Playback" if playback_mode else "Normal")

BASE_URL = 'http://127.0.0.1:5000'
def push_data(data):
    try:
        r = requests.post(f"{BASE_URL}/api/update", json=data)
    except Exception as e:
        pass

def data_init():
    init_data = {
        'config': default_config,
        'step': '',
        'potential_actions': '',        
        'temperature': '',     
        'decision': '',    
        'action_goal': {},
        'suspend_actions': [],
        'candidate_actions': [],
        'selected_action_id': None,
        'result_tree': {},
        'delete_ids': [],
        'exec_chain': [],           # merged: screen, operation
        'explore_tree': [],         # merged: name, state, children
        'result': ''                # new: to store the result of operations
    }
    push_data(init_data)

def record_data(record):
    if not os.path.exists(RECORD_DIR):
        os.makedirs(RECORD_DIR)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    record_path = os.path.join(RECORD_DIR, f"{now}.json")
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ---- Run Server ----
if __name__ == '__main__':
    server.run(debug=True)
