import yaml
import json
import re
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify
from dash import Dash, html, dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State, ALL, MATCH
import requests
import pyautogui
import datetime
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BottomUpAgent.Eye import Eye
from BottomUpAgent.LongMemory import LongMemory
from BottomUpAgent.Hand import Hand
from dash import ctx
import threading
import time
import re

def highlight_text(text, search_term):
    """
    Highlight matching search terms in text
    Returns a list of Dash components with highlighted elements
    """
    if not search_term or not text:
        return [text]
    
    # Escape special characters and create case-insensitive regex
    escaped_term = re.escape(search_term)
    pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
    
    # Split text
    parts = pattern.split(text)
    
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # Odd indices are matched parts
            # Highlight matched text
            result.append(html.Span(
                part,
                style={
                    'background-color': '#FFA500',  # Orange background
                    # 'color': 'white',
                    'padding': '1px 2px',
                    'border-radius': '2px',
                    'font-weight': 'bold'
                }
            ))
        else:
            # Regular text
            if part:  # Only add non-empty strings
                result.append(part)
    
    return result

def load_config(config_file=None):
    """Load configuration from specified file or default"""
    if config_file is None or not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}") 
    
    # Handle relative paths - make them relative to project root
    if not os.path.isabs(config_file):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_file)
    else:
        config_path = config_file
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ---- Parse command line arguments ----
def parse_args():
    parser = argparse.ArgumentParser(description='Bottom-Up Agent Visualizer')
    parser.add_argument('--config_file', type=str, default=None, help='Path to configuration YAML file (default: config/sts_explore_claude.yaml)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind the server (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind the server (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

# ---- Global variables ----
default_config = None
eye = None
hand = None
global_long_memory = None

def initialize_components(config_file=None):
    """Initialize components with configuration"""
    global default_config, eye, hand, global_long_memory
    
    if default_config is None:
        default_config = load_config(config_file)
        
    if eye is None:
        eye = Eye(default_config)
        
    if hand is None:
        hand = Hand(default_config)
        
    if global_long_memory is None:
        try:
            global_long_memory = LongMemory(default_config)
            print("Global LongMemory initialized successfully")
        except Exception as e:
            print(f"Error creating global LongMemory instance: {e}")
            global_long_memory = None

def get_long_memory():
    """Return the global LongMemory instance to avoid repeated initialization"""
    return global_long_memory

def highlight_yaml(yaml_str):
    """Convert YAML string to highlighted HTML with Python-based syntax highlighting"""
    if not yaml_str:
        return html.Div("No configuration available", 
                       style={'text-align': 'center', 'color': '#999', 'padding': '20px'})
    
    import re
    
    # Split into lines for processing
    lines = yaml_str.split('\n')
    highlighted_lines = []
    
    for line in lines:
        if not line.strip():
            highlighted_lines.append(html.Br())
            continue
            
        # Create spans for different YAML elements
        line_elements = []
        
        # Check for comments
        if '#' in line:
            parts = line.split('#', 1)
            main_part = parts[0]
            comment_part = '#' + parts[1] if len(parts) > 1 else ''
        else:
            main_part = line
            comment_part = ''
        
        # Process main part
        if ':' in main_part and not main_part.strip().startswith('-'):
            # Key-value pair
            key_value = main_part.split(':', 1)
            key_part = key_value[0]
            value_part = ':' + key_value[1] if len(key_value) > 1 else ''
            
            # Add indentation
            indent_match = re.match(r'^(\s*)', key_part)
            if indent_match:
                line_elements.append(html.Span(indent_match.group(1)))
                key_part = key_part[len(indent_match.group(1)):]
            
            # Add key (colored)
            line_elements.append(html.Span(key_part, style={'color': '#667eea', 'font-weight': '600'}))
            
            # Add colon and value
            if value_part:
                colon_and_value = value_part
                if colon_and_value.strip() == ':':
                    line_elements.append(html.Span(':', style={'color': '#2d3748'}))
                else:
                    # Split colon and value
                    line_elements.append(html.Span(':', style={'color': '#2d3748'}))
                    value_text = colon_and_value[1:].strip()
                    if value_text:
                        # Determine value type and color
                        if value_text.lower() in ['true', 'false']:
                            color = '#e53e3e'  # Boolean
                        elif value_text.lower() in ['null', 'none', '~']:
                            color = '#805ad5'  # Null
                        elif re.match(r'^-?\d+\.?\d*$', value_text):
                            color = '#d69e2e'  # Number
                        elif value_text.startswith('"') and value_text.endswith('"'):
                            color = '#38a169'  # Quoted string
                        elif value_text.startswith("'") and value_text.endswith("'"):
                            color = '#38a169'  # Quoted string
                        else:
                            color = '#38a169'  # Unquoted string
                        
                        line_elements.append(html.Span(' '))
                        line_elements.append(html.Span(value_text, style={'color': color, 'font-weight': '500'}))
        else:
            # List item or other
            if main_part.strip().startswith('-'):
                # List item
                indent_match = re.match(r'^(\s*)', main_part)
                if indent_match:
                    line_elements.append(html.Span(indent_match.group(1)))
                    main_part = main_part[len(indent_match.group(1)):]
                
                line_elements.append(html.Span('-', style={'color': '#3182ce', 'font-weight': '600'}))
                rest = main_part[1:]
                if rest:
                    line_elements.append(html.Span(rest, style={'color': '#2d3748'}))
            else:
                # Regular text
                line_elements.append(html.Span(main_part, style={'color': '#2d3748'}))
        
        # Add comment if exists
        if comment_part:
            line_elements.append(html.Span(comment_part, style={'color': '#a0aec0', 'font-style': 'italic'}))
        
        # Wrap line in div
        highlighted_lines.append(html.Div(line_elements, style={'margin': '0', 'line-height': '1.6'}))
    
    return html.Div(
        className='enhanced-code-container',
        children=[
            html.Pre(
                highlighted_lines,
                style={
                    'font-family': 'JetBrains Mono, Monaco, Consolas, "Courier New", monospace',
                    'font-size': '13px',
                    'line-height': '1.6',
                    'margin': '2px',
                    'padding': '20px',
                    'background': 'rgba(255, 255, 255, 0.95)',
                    'backdrop-filter': 'blur(10px)',
                    'border-radius': '10px',
                    'overflow-x': 'auto'
                }
            )
        ],
        style={
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'border-radius': '12px',
            'padding': '2px',
            'box-shadow': '0 8px 32px rgba(0, 0, 0, 0.1)',
            'margin': '10px 0'
        }
    )

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
    'result': '',               # new: to store the result of operations
    'skills': [],               # skills list
    'skill_clusters': []        # skill clusters
}

# Initialize RECORD_DIR safely
RECORD_DIR = None
if default_config:
    RECORD_DIR = os.path.join(os.getcwd(), default_config['result_path'], 
                              default_config['game_name'], default_config['run_name'])
else:
    RECORD_DIR = os.path.join(os.getcwd(), 'results', 'default_game', 'default_run')

recording = False

# Playback states
playback_mode = False
playback_folder = ''
playback_files = []
playback_index = 0


# ---- Flask + Dash Setup ----
server = Flask(__name__)
app = Dash(__name__, server=server, 
    external_stylesheets=[
        '/static/styles.css',
        # Modern CSS Frameworks
        'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
        'https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css',
        'https://unpkg.com/purecss@3.0.0/build/pure-min.css',
        # Icon Libraries
        'https://cdn.jsdelivr.net/npm/lucide@latest/dist/umd/lucide.css',
        'https://cdn.jsdelivr.net/npm/heroicons@2.0.18/24/outline/index.css',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',

        # Additional Modern UI Libraries
        'https://unpkg.com/alpinejs@3.10.5/dist/cdn.min.js',
    ],
    external_scripts=[]
)

# ---- API Endpoint for Updates ----
@server.route('/api/update', methods=['POST'])
def update_data():
    data = request.get_json()
    for key, value in data.items():
        if key in global_data:
            global_data[key] = value
    return {'status': 'ok'}

# ---- Static CSS File Route ----
@server.route('/static/styles.css')
def serve_css():
    from flask import send_from_directory
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, 'styles.css')


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
    html.H1("Bottom-Up Agent Visualizer"),

    # Modern button styles with Font Awesome icons
    html.Div(className='button-container', children=[
        html.Button([
            html.I(className='fas fa-play', style={'margin-right': '8px'}),
            'Start Record'
        ], id='start-record', className='modern-button btn-start-record'),
        html.Button([
            html.I(className='fas fa-stop', style={'margin-right': '8px'}),
            'Stop Record'
        ], id='stop-record', className='modern-button btn-stop-record'),
        html.Button([
            html.I(className='fas fa-video', style={'margin-right': '8px'}),
            'Start Play'
        ], id='start-playback', className='modern-button btn-start-play'),
        html.Button([
            html.I(className='fas fa-pause', style={'margin-right': '8px'}),
            'Stop Play'
        ], id='stop-playback', className='modern-button btn-stop-play'),
        dcc.Input(id='playback-folder', type='text', placeholder='Input folder name', className='modern-input', style={'width': '200px'}),
        html.Div(id='record-status', style={
            'align-self': 'center', 
            'font-weight': '600',
            'color': '#2d3748',
            'font-size': '14px'
        })
    ]),

    # Top row: Interface, Run States & Actions, Explore Tree
    html.Div(className='main-row', children=[
        # Game Interface
        html.Div(className='game-interface-section', children=[
            html.H2([
                html.I(className='fas fa-gamepad', style={'margin-right': '8px'}),
                'Game Interface'
            ]),
            html.Img(id='game-image'),
        ]),

        # Run States, Action Goal
        html.Div(className='run-states-section', children=[
            html.H2([
                html.I(className='fas fa-running', style={'margin-right': '8px'}),
                'Run States'
            ]),
            html.Div(id='run-states', style={'display': 'flex', 'gap': '20px', 'flex-wrap': 'wrap'}),
            html.H2([
                html.I(className='fas fa-bullseye', style={'margin-right': '8px'}),
                'Action Goal'
            ]),
            html.Div(id='action-goal'),
        ]),

        # Skills Panel
        html.Div(className='skills-section', children=[
            html.H2([
                html.I(className='fas fa-brain', style={'margin-right': '8px'}),
                'Skills Library'
            ]),
            html.Div(className='skills-search-container', style={
                'display': 'flex', 
                'gap': '12px', 
                'margin-bottom': '10px', 
                'align-items': 'center',
                'width': '100%'
            }, children=[
                html.Button([
                    html.I(className='fas fa-sync-alt')
                ], id='refresh-skills', className='modern-button btn-refresh', style={
                    'height': '48px',
                    'width': '48px',
                    'min-width': '48px',
                    'flex-shrink': '0'
                }),
                dcc.Dropdown(
                    id='skill-filter',
                    options=[
                        {'label': 'All Skills', 'value': 'all'},
                        {'label': 'High Fitness (>2)', 'value': 'high_fitness'},
                        {'label': 'Frequently Used (>5)', 'value': 'frequent'},
                        {'label': 'Recent', 'value': 'recent'}
                    ],
                    value='all',
                    className='modern-dropdown skills-filter-dropdown',
                    style={
                        'width': '200px', 
                        'min-width': '200px',
                        'flex-shrink': '0'
                    }
                ),
                dcc.Input(
                    id='skill-search',
                    type='text',
                    placeholder='Search skills by name or description...',
                    value='',
                    className='modern-input skills-search-input',
                    style={
                        'flex': '1',
                        'min-width': '250px',
                        'height': '48px'
                    }
                )
            ]),
            html.Div(id='skills-count', style={'margin-bottom': '10px', 'font-size': '14px', 'color': '#666'}),
            html.Div(id='skills-list', style={
                'height': '400px', 
                'overflow-y': 'auto', 
                'border': '1px solid #ddd', 
                'padding': '10px',
                'background': '#f9f9f9'
            })
        ])
    ]),

    # Second row: Candidate Actions & Exec Chain + Explore Tree
    html.Div(className='content-row', children=[
        # Candidate Actions 
        html.Div(className='content-section', children=[
            html.H2([
                html.I(className='fas fa-clipboard-list', style={'margin-right': '8px'}),
                'Candidate Actions'
            ]),
            html.Div(id='candidates', style={'display': 'flex', 'flex-wrap': 'wrap'})
        ]),

        # Explore Tree
        html.Div(className='content-section', children=[
            html.H2([
                html.I(className='fas fa-sitemap', style={'margin-right': '8px'}),
                'Explore Tree'
            ]),
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

    # Third row: Exec Chain + LLM + Results
    html.Div(className='content-row', children=[
        html.Div(className='content-section', children=[
            html.H2([
                html.I(className='fas fa-arrow-right', style={'margin-right': '8px'}),
                'Exec Chain & LLM'
            ]),
            html.Div(id='exec-chain', style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'center'}),
            html.H2([
                html.I(className='fas fa-file-alt', style={'margin-right': '8px'}),
                'Result'
            ]),
            html.Pre(id='result-text', style={'whiteSpace': 'pre-wrap'})
        ]),
    ]),

    # Fourth row: config
    html.Div(
        children=[
            html.H2([
                html.I(className='fas fa-cog', style={'margin-right': '8px'}),
                'Config'
            ]),
            html.Div(id='config')
        ]
    ),

    # Hidden divs for storing state
    html.Div(id='click-mode-state', style={'display': 'none'}, children='left'),
    html.Div(id='dummy-output', style={'display': 'none'}),
    

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
    Output('skills-list', 'children'),
    Output('skills-count', 'children'),
    Input('interval', 'n_intervals'),
    Input('refresh-skills', 'n_clicks'),
    State('skill-filter', 'value'),
    State('skill-search', 'value')
)
def update_ui(n, refresh_clicks, skill_filter, search_text):
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

    # 2. Config - convert to YAML for better readability
    try:
        config_yaml = yaml.dump(global_data['config'], default_flow_style=False, allow_unicode=True, indent=2)
        config_display = highlight_yaml(config_yaml)
    except Exception as e:
        config_display = html.Pre(f"Error displaying config: {str(e)}", style={'color': 'red'})

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

    # 9. Skills List
    try:
        long_memory = get_long_memory()
        if long_memory is not None:
            all_skills = long_memory.get_skills()
            total_skills_count = len(all_skills)
            skills = all_skills
            skill_filter = skill_filter or 'all'
            
            # Filter skills based on selection
            if skill_filter == 'high_fitness':
                skills = [s for s in skills if s.get('fitness', 0) > 2]
            elif skill_filter == 'frequent':
                skills = [s for s in skills if s.get('num', 0) > 5]
            elif skill_filter == 'recent':
                skills = sorted(skills, key=lambda x: x.get('id', 0), reverse=True)[:20]
            
            # Apply search filter if search text is provided
            if search_text and search_text.strip():
                search_term = search_text.strip().lower()
                filtered_skills = []
                for skill in skills:
                    # Search in skill name
                    if search_term in skill.get('name', '').lower():
                        filtered_skills.append(skill)
                        continue
                    
                    # Search in skill description
                    if search_term in skill.get('description', '').lower():
                        filtered_skills.append(skill)
                        continue
                
                skills = filtered_skills
            
            filtered_count = len(skills)
            search_info = f" | Search: '{search_text}'" if search_text and search_text.strip() else ""
            skills_count_text = f"📊 Total: {total_skills_count} skills | Displayed: {filtered_count} skills{search_info}"
            
            skills_div = []
            # Display all skills (removed 50 limit) - you can add limit back if performance issues occur
            for skill in skills:
                fitness_color = '#4CAF50' if skill.get('fitness', 0) > 2 else '#FF9800' if skill.get('fitness', 0) > 1 else '#F44336'
                usage_color = '#2196F3' if skill.get('num', 0) > 5 else '#9E9E9E'
                
                skill_card = html.Div([
                    # ID Badge in top-left corner
                    html.Div(
                        str(skill.get('id', '?')),
                        style={
                            'position': 'absolute',
                            'top': '-8px',
                            'left': '-8px',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'color': 'white',
                            'width': '24px',
                            'height': '24px',
                            'border-radius': '50%',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center',
                            'font-size': '11px',
                            'font-weight': 'bold',
                            'box-shadow': '0 2px 4px rgba(0,0,0,0.2)',
                            'border': '2px solid white',
                            'z-index': '10'
                        }
                    ),
                    # Copy JSON Button in top-right corner
                    html.Button([
                        html.I(className='fa fa-clipboard copy-icon'),
                        html.Span('Copy JSON', className='copy-text')
                    ],
                        id={'type': 'copy-skill-btn', 'index': skill.get('id', 'unknown')},
                        className='copy-json-btn',
                        title="Copy JSON"
                    ),
                    # Hidden div containing the JSON data for copying
                    html.Div(
                        json.dumps(skill, indent=2, ensure_ascii=False),
                        id={'type': 'skill-json-data', 'index': skill.get('id', 'unknown')},
                        style={'display': 'none'}
                    ),
                    html.Div([
                        # Title and status badges in one row
                        html.Div([
                            html.H4(
                                highlight_text(skill.get('name', f"Skill {skill.get('id', 'Unknown')}"), search_text.strip() if search_text else None), 
                                style={'margin': '0', 'color': '#333', 'font-size': '14px', 'padding-left': '10px', 'flex': '1'}
                            ),
                            html.Div([
                                html.Span(f"Fitness: {skill.get('fitness', 0):.2f}", 
                                        style={'background': fitness_color, 'color': 'white', 'padding': '2px 6px', 
                                              'border-radius': '3px', 'font-size': '10px', 'margin-right': '5px'}),
                                html.Span(f"Used: {skill.get('num', 0)}x", 
                                        style={'background': usage_color, 'color': 'white', 'padding': '2px 6px', 
                                              'border-radius': '3px', 'font-size': '10px'}),
                            ], style={'display': 'flex', 'align-items': 'center', 'gap': '0px'})
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'margin-bottom': '8px'}),
                        html.P(
                            highlight_text(skill.get('description', 'No description'), search_text.strip() if search_text else None), 
                            style={'margin': '0 0 10px 0', 'font-size': '12px', 'color': '#666', 'line-height': '1.3'}
                        ),
                        html.Details([
                            html.Summary("Operations", style={'font-size': '11px', 'cursor': 'pointer'}),
                            html.Pre(json.dumps(skill.get('operations', []), indent=1), 
                                   style={'font-size': '10px', 'background': '#f5f5f5', 'padding': '5px', 
                                         'border-radius': '3px', 'max-height': '100px', 'overflow-y': 'auto'})
                        ])
                    ])
                ], className='skill-card', style={
                    'position': 'relative',
                    'border': '1px solid #ddd', 
                    'border-radius': '8px', 
                    'padding': '12px', 
                    'margin': '8px 0',
                    'background': 'white',
                    'box-shadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'transition': 'transform 0.2s ease, box-shadow 0.2s ease',
                    'overflow': 'visible'
                })
                skills_div.append(skill_card)
            
            if not skills_div:
                skills_div = [html.Div("No skills found", style={'text-align': 'center', 'color': '#999', 'padding': '20px'})]
                skills_count_text = f"📊 Total: {total_skills_count} skills | Displayed: 0 skills"
        else:
            skills_div = [html.Div("LongMemory not available", style={'text-align': 'center', 'color': '#999', 'padding': '20px'})]
            skills_count_text = "❌ LongMemory not available"
            
    except Exception as e:
        skills_div = [html.Div(f"Error loading skills: {str(e)}", style={'color': 'red', 'padding': '10px'})]
        skills_count_text = f"❌ Error loading skills: {str(e)}"

    # 10. Save conditionally
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

    return screenshot, config_display, ag_div, run_states_div, cand, explore_elements, exec_div, result_txt, skills_div, skills_count_text

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

# ---- Callback for Skills Search ----
@app.callback(
    Output('skills-list', 'children', allow_duplicate=True),
    Output('skills-count', 'children', allow_duplicate=True),
    Input('skill-search', 'value'),
    State('skill-filter', 'value'),
    prevent_initial_call=True
)
def update_skills_search(search_text, skill_filter):
    """Update skills list based on search input"""
    try:
        long_memory = get_long_memory()
        if long_memory is not None:
            all_skills = long_memory.get_skills()
            total_skills_count = len(all_skills)
            skills = all_skills
            skill_filter = skill_filter or 'all'
            
            # Filter skills based on selection
            if skill_filter == 'high_fitness':
                skills = [s for s in skills if s.get('fitness', 0) > 2]
            elif skill_filter == 'frequent':
                skills = [s for s in skills if s.get('num', 0) > 5]
            elif skill_filter == 'recent':
                skills = sorted(skills, key=lambda x: x.get('id', 0), reverse=True)[:20]
            
            # Apply search filter if search text is provided
            if search_text and search_text.strip():
                search_term = search_text.strip().lower()
                filtered_skills = []
                for skill in skills:
                    # Search in skill name
                    if search_term in skill.get('name', '').lower():
                        filtered_skills.append(skill)
                        continue
                    
                    # Search in skill description
                    if search_term in skill.get('description', '').lower():
                        filtered_skills.append(skill)
                        continue
                
                skills = filtered_skills
            
            filtered_count = len(skills)
            search_info = f" | Search: '{search_text}'" if search_text and search_text.strip() else ""
            skills_count_text = f"📊 Total: {total_skills_count} skills | Displayed: {filtered_count} skills{search_info}"
            
            skills_div = []
            # Display all skills
            for skill in skills:
                fitness_color = '#4CAF50' if skill.get('fitness', 0) > 2 else '#FF9800' if skill.get('fitness', 0) > 1 else '#F44336'
                usage_color = '#2196F3' if skill.get('num', 0) > 5 else '#9E9E9E'
                
                skill_card = html.Div([
                    # ID Badge in top-left corner
                    html.Div(
                        str(skill.get('id', '?')),
                        style={
                            'position': 'absolute',
                            'top': '-8px',
                            'left': '-8px',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'color': 'white',
                            'width': '24px',
                            'height': '24px',
                            'border-radius': '50%',
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center',
                            'font-size': '11px',
                            'font-weight': 'bold',
                            'box-shadow': '0 2px 4px rgba(0,0,0,0.2)',
                            'border': '2px solid white',
                            'z-index': '10'
                        }
                    ),
                    # Copy JSON Button in top-right corner
                    html.Button([
                        html.I(className='fa fa-clipboard copy-icon'),
                        html.Span('Copy JSON', className='copy-text')
                    ],
                        id={'type': 'copy-skill-btn', 'index': skill.get('id', 'unknown')},
                        className='copy-json-btn',
                        title="Copy JSON"
                    ),
                    # Hidden div containing the JSON data for copying
                    html.Div(
                        json.dumps(skill, indent=2, ensure_ascii=False),
                        id={'type': 'skill-json-data', 'index': skill.get('id', 'unknown')},
                        style={'display': 'none'}
                    ),
                    html.Div([
                        # Title and status badges in one row
                        html.Div([
                            html.H4(skill.get('name', f"Skill {skill.get('id', 'Unknown')}"), 
                                   style={'margin': '0', 'color': '#333', 'font-size': '14px', 'padding-left': '10px', 'flex': '1'}),
                            html.Div([
                                html.Span(f"Fitness: {skill.get('fitness', 0):.2f}", 
                                        style={'background': fitness_color, 'color': 'white', 'padding': '2px 6px', 
                                              'border-radius': '3px', 'font-size': '10px', 'margin-right': '5px'}),
                                html.Span(f"Used: {skill.get('num', 0)}x", 
                                        style={'background': usage_color, 'color': 'white', 'padding': '2px 6px', 
                                              'border-radius': '3px', 'font-size': '10px'}),
                            ], style={'display': 'flex', 'align-items': 'center', 'gap': '0px'})
                        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'margin-bottom': '8px'}),
                        html.P(skill.get('description', 'No description'), 
                              style={'margin': '0 0 10px 0', 'font-size': '12px', 'color': '#666', 'line-height': '1.3'}),
                        html.Details([
                            html.Summary("Operations", style={'font-size': '11px', 'cursor': 'pointer'}),
                            html.Pre(json.dumps(skill.get('operations', []), indent=1), 
                                   style={'font-size': '10px', 'background': '#f5f5f5', 'padding': '5px', 
                                         'border-radius': '3px', 'max-height': '100px', 'overflow-y': 'auto'})
                        ])
                    ])
                ], className='skill-card', style={
                    'position': 'relative',
                    'border': '1px solid #ddd', 
                    'border-radius': '8px', 
                    'padding': '12px', 
                    'margin': '8px 0',
                    'background': 'white',
                    'box-shadow': '0 2px 6px rgba(0,0,0,0.1)',
                    'transition': 'transform 0.2s ease, box-shadow 0.2s ease',
                    'overflow': 'visible'
                })
                skills_div.append(skill_card)
            
            if not skills_div:
                skills_div = [html.Div("No skills found", style={'text-align': 'center', 'color': '#999', 'padding': '20px'})]
                skills_count_text = f"📊 Total: {total_skills_count} skills | Displayed: 0 skills{search_info}"
                
            return skills_div, skills_count_text
        else:
            skills_div = [html.Div("LongMemory not available", style={'text-align': 'center', 'color': '#999', 'padding': '20px'})]
            skills_count_text = "❌ LongMemory not available"
            return skills_div, skills_count_text
            
    except Exception as e:
        skills_div = [html.Div(f"Error loading skills: {str(e)}", style={'color': 'red', 'padding': '10px'})]
        skills_count_text = f"❌ Error loading skills: {str(e)}"
        return skills_div, skills_count_text

# ---- Callback for Copy Skill JSON Button ----
@app.callback(
    Output({'type': 'skill-json-data', 'index': MATCH}, 'children'),
    Input({'type': 'copy-skill-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def update_skill_json_for_copy(n_clicks):
    """Update the hidden JSON data when copy button is clicked"""
    if not n_clicks:
        return ""
    
    # Get the triggered button
    triggered = ctx.triggered_id
    if triggered and triggered.get('type') == 'copy-skill-btn':
        skill_id = triggered.get('index')
        
        # Find the skill data
        try:
            if hasattr(global_data.get('LongMemory'), 'skills'):
                skills = global_data['LongMemory'].skills
                for skill in skills:
                    if str(skill.get('id')) == str(skill_id):
                        # Create a clean copy of the skill data
                        return json.dumps(skill, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error preparing skill JSON: {e}")
    
    return ""

# ---- Client-side callback for copying to clipboard ----
app.clientside_callback(
    """
    function(n_clicks_list, json_data_list) {
        // Check if any button was clicked
        if (!n_clicks_list || !json_data_list) {
            return '';
        }
        
        // Find which button was clicked
        let clickedIndex = -1;
        for (let i = 0; i < n_clicks_list.length; i++) {
            if (n_clicks_list[i] && n_clicks_list[i] > 0) {
                clickedIndex = i;
                break;
            }
        }
        
        if (clickedIndex >= 0 && json_data_list[clickedIndex]) {
            const jsonData = json_data_list[clickedIndex];
            
            // Ensure document is focused before copying
            window.focus();
            document.body.focus();
            
            // Try multiple copy methods
            function copyToClipboard(text) {
                // Method 1: Modern clipboard API
                if (navigator.clipboard && window.isSecureContext) {
                    return navigator.clipboard.writeText(text).then(() => {
                        console.log('JSON copied to clipboard!');
                        return true;
                    }).catch(err => {
                        console.warn('Clipboard API failed:', err);
                        return false;
                    });
                }
                
                // Method 2: Fallback using textarea
                return new Promise((resolve) => {
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.left = '-999999px';
                    textarea.style.top = '-999999px';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    
                    try {
                        const successful = document.execCommand('copy');
                        if (successful) {
                            console.log('JSON copied to clipboard!');
                            resolve(true);
                        } else {
                            resolve(false);
                        }
                    } catch (err) {
                        console.warn('execCommand failed:', err);
                        resolve(false);
                    } finally {
                        document.body.removeChild(textarea);
                    }
                });
            }
            
            // Perform copy and show feedback
            copyToClipboard(jsonData).then(success => {
                if (success) {
                    // Find and update the clicked button
                    const buttons = document.querySelectorAll('[id*="copy-skill-btn"]');
                    if (buttons[clickedIndex]) {
                        const btn = buttons[clickedIndex];
                        const icon = btn.querySelector('.copy-icon');
                        
                        // Add clicked class for CSS animation
                        btn.classList.add('clicked');
                        
                        // Change icon to checkmark
                        if (icon) {
                            const originalIcon = icon.className;
                            icon.className = 'fa fa-check copy-icon';
                            
                            // Restore original state after animation
                            setTimeout(() => {
                                btn.classList.remove('clicked');
                                icon.className = originalIcon;
                            }, 1500);
                        }
                    }
                } else {
                    alert('Copy failed, please redo manually with the following text!\\n\\n' + jsonData.substring(0, 200) + '...');
                }
            });
        }
        
        return '';
    }
    """,
    Output('dummy-output', 'children', allow_duplicate=True),
    Input({'type': 'copy-skill-btn', 'index': ALL}, 'n_clicks'),
    State({'type': 'skill-json-data', 'index': ALL}, 'children'),
    prevent_initial_call=True
)

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
        'result': '',               # new: to store the result of operations
        'skills': [],               # skills list
        'skill_clusters': []        # skill clusters
    }
    push_data(init_data)

def record_data(record):
    if not os.path.exists(RECORD_DIR):
        os.makedirs(RECORD_DIR)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    record_path = os.path.join(RECORD_DIR, f"{now}.json")
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def main():
    """Main entry point for the visualizer"""
    # Parse arguments and initialize components
    args = parse_args()
    initialize_components(args.config_file)
    
    print("Starting Enhanced Visualizer...")
    print(f"Configuration file: {args.config_file or 'config/sts_explore_claude.yaml'}")
    print("Features:")
    print("- 🧠 Skills Library with filtering")
    print("- 🖱️ Interactive mouse control (click on game image)")
    print("- ⌨️ Keyboard input support")
    print("- 🎮 Real-time game monitoring")
    print(f"\nAccess the visualizer at: http://{args.host}:{args.port}")
    server.run(host=args.host, port=args.port, debug=args.debug)

# ---- Run Server ----
if __name__ == '__main__':
    main()
