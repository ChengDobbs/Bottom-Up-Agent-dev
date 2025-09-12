import yaml
import os

def load_mcp_tools_config(config_path=None):
    """
    Load MCP tools configuration from config file
    """
    if config_path is None:
        # Default to crafter config, can be overridden
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'gym', 'crafter_config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('mcp_tools', {})
    except Exception as e:
        print(f"Warning: Could not load MCP tools config from {config_path}: {e}")
        return {}

def get_enabled_tools_for_mode(config, mode='full_mode'):
    """
    Get enabled tools for a specific mode from config
    """
    if not config:
        return None
    
    mode_config = config.get(mode, {})
    enabled_tools = mode_config.get('enabled_tools', [])
    
    if enabled_tools == 'all':
        # Return all available tools
        all_tools = []
        all_tools.extend(config.get('environment_tools', []))
        all_tools.extend(config.get('skill_tools', []))
        all_tools.extend(config.get('operation_tools', []))
        all_tools.extend(config.get('game_specific_tools', []))
        return all_tools
    
    return enabled_tools

def generate_skill_tools(model_name):
    if "claude" in model_name:
        return [
            {
                "name": "save_skill",
                "description": "Save the skill to long memory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the skill"
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the skill"
                        }
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False
                }
            },
            {
                "name": "no_meaning_skill",
                "description": "The skill is meaningless",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "incomplete_skill",
                "description": "The skill represents an incomplete operation sequence that requires continuation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the incomplete skill"
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of what was done and what needs to continue"
                        },
                        "next_action_hint": {
                            "type": "string",
                            "description": "Hint about what action should be performed next to complete the sequence"
                        }
                    },
                    "required": ["name", "description", "next_action_hint"],
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "save_skill",
                    "description": "Save the skill to long memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the skill"
                            },
                            "description": {
                                "type": "string",
                                "description": "The description of the skill"
                            }
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False
                    }
                }
            }
        ]
    else:
        print(f"Model {model_name} not found")
        return None


def environment_query_tools(model_name):
    """Environment query tools for MCP-style interaction"""
    if "claude" in model_name:
        return [
            {
                "name": "query_environment",
                "description": "Query current UI environment information to get real-time data about detected objects",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["all_objects", "text_content", "specific_object", "object_summary", "game_state", "historical_context", "scene_summary", "available_actions"],
                            "description": "Type of environment query to perform. 'scene_summary' provides concise player surroundings info. 'available_actions' returns prioritized actions player can execute now."
                        },
                        "object_id": {
                            "type": "string",
                            "description": "Specific object ID to query (required for specific_object query_type)"
                        },
                        "context_query": {
                            "type": "string",
                            "description": "Query string for historical context search (optional for historical_context query_type)"
                        }
                    },
                    "required": ["query_type"],
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_environment",
                    "description": "Query current UI environment information to get real-time data about detected objects",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": ["all_objects", "text_content", "specific_object", "object_summary", "game_state", "historical_context", "scene_summary", "available_actions"],
                                "description": "Type of environment query to perform. 'scene_summary' provides concise player surroundings info. 'available_actions' returns prioritized actions player can execute now. Note: 'clickable_objects' has been replaced with 'historical_context' for better efficiency"
                            },
                            "object_id": {
                                "type": "string",
                                "description": "Specific object ID to query (required for specific_object query_type)"
                            },
                            "context_query": {
                                "type": "string",
                                "description": "Query string for historical context search (optional for historical_context query_type)"
                            }
                        },
                        "required": ["query_type"],
                        "additionalProperties": False
                    }
                }
            }
        ]
    else:
        print(f"Model {model_name} not found")
        return None


def mcp_interaction_tools(model_name, config_path=None, mode='full_mode'):
    """Combined tools for MCP-style structured interaction with configurable tool selection"""
    # Load MCP tools configuration
    mcp_config = load_mcp_tools_config(config_path)
    enabled_tools = get_enabled_tools_for_mode(mcp_config, mode)
    
    # Get all available tool functions
    all_tools = []
    
    # Add environment tools if enabled
    if not enabled_tools or any(tool in enabled_tools for tool in ['query_environment', 'query_hand', 'query_energy', 'query_health', 'query_enemies']):
        env_tools = environment_query_tools(model_name)
        if env_tools:
            all_tools.extend(env_tools)
    
    # Add skill tools if enabled
    if not enabled_tools or any(tool in enabled_tools for tool in ['select_skill', 'evaluate_combo']):
        skill_tools = select_skill_tools(model_name)
        if skill_tools:
            all_tools.extend(skill_tools)
    
    # Add operation tools if enabled
    if not enabled_tools or any(tool in enabled_tools for tool in ['Click', 'RightSingle', 'LeftDouble', 'Type', 'Drag', 'Finished', 'Scroll']):
        operation_tools = do_operation_tools(model_name)
        if operation_tools:
            all_tools.extend(operation_tools)
    
    # Add keyboard action tools if enabled
    if not enabled_tools or any(tool in enabled_tools for tool in ['move_actions', 'craft_actions', 'interact_actions']):
        keyboard_tools = keyboard_action_tools(model_name)
        if keyboard_tools:
            all_tools.extend(keyboard_tools)
    
    # Log which tools are being used
    if mcp_config and enabled_tools:
        print(f"MCP Tools loaded for mode '{mode}': {enabled_tools}")
    else:
        print(f"MCP Tools: Using default full tool set (no config found or full_mode)")
    
    return all_tools if all_tools else None


def keyboard_action_tools(model_name):
    """Generic keyboard action tools for game control - specific mappings defined in config"""
    if "claude" in model_name:
        return [
            {
                "name": "move_up",
                "description": "Move character/cursor up",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "move_down",
                "description": "Move character/cursor down",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "move_left",
                "description": "Move character/cursor left",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "move_right",
                "description": "Move character/cursor right",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "interact",
                "description": "Primary interaction action",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "sleep",
                "description": "Rest/sleep action",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "place_table",
                "description": "Place crafting table",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "place_stone",
                "description": "Place stone block",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "place_furnace",
                "description": "Place furnace",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "plant_sapling",
                "description": "Plant sapling",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_wood_pickaxe",
                "description": "Craft wood pickaxe",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_stone_pickaxe",
                "description": "Craft stone pickaxe",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_iron_pickaxe",
                "description": "Craft iron pickaxe",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_wood_sword",
                "description": "Craft wood sword",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_stone_sword",
                "description": "Craft stone sword",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },
            {
                "name": "craft_iron_sword",
                "description": "Craft iron sword",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        # GPT-4 format tools would go here
        return None
    else:
        print(f"Model {model_name} not found")
        return None


def get_explore_guidance_tools(model_name):

    if "claude" in model_name:
        return [
            {
                "name": "LeftSingleClick",
                "description": "Click at the given coordinates(select the item)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "RightSingleClick",
                "description": "Right click at the given coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "LeftDoubleClick",
                "description": "Double click at the given coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "TypeText",
                "description": "Type the given text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"],
                    "additionalProperties": False
                },
            },
            {
                "name": "Drag",
                "description": "Drag from the first coordinates to the second coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"}
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                    "additionalProperties": False
                },
            },
            {
                "name": "Finished",
                "description": "Finish the task",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
            }
        ]

    elif model_name == "gpt-4o" or model_name == "o4-mini":
        schema = "parameters"
        return None
    else:
        print(f"Model {model_name} not found")
        return None
    
def select_skill_tools(model_name):
    if "claude" in model_name:
        return [
            {
                "name": "select_skill",
                "description": "Select the best skill from the candidate skills",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "number"}
                    },
                    "required": ["id"],
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "select_skill",
                    "description": "Select the best skill from the candidate skills",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "number"}
                        },
                        "required": ["id"],
                        "additionalProperties": False   
                    }
                }
            }
        ]
    else:
        print(f"Model {model_name} not found")
        return None
    
def skill_evaluate_tools(model_name):
    if "claude" in model_name:
        return [
            {
                "name": "skill_evaluate",
                "description": "Analyze the skill and give me a evaluation on consistency and progressiveness",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "is_consistent": {
                            "type": "boolean",
                            "description": "Whether the skill matches its description and intent"
                        },
                        "is_progressive": {
                            "type": "boolean",
                            "description": "Whether the skill made real progress"
                        }
                    },
                    "required": ["is_consistent", "is_progressive"],
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "skill_evaluate",
                    "description": "Analyze the skill and give me a evaluation on consistency and progressiveness",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_consistent": {
                                "type": "boolean",
                                "description": "Whether the skill matches its description and intent"
                            },
                            "is_progressive": {
                                "type": "boolean",
                                "description": "Whether the skill made real progress"
                            },
                        },
                        "required": ["is_consistent", "is_progressive"],
                        "additionalProperties": False
                    }
                }
            }       
        ]
    else:
        print(f"Model {model_name} not found")
        return None 


def skill_evaluate2_tools(model_name):
    if "claude" in model_name:
        return [
            {
                "name": "skill_evaluate",
                "description": "Analyze the skill and give me a evaluation on progressiveness",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "is_progressive": {
                            "type": "boolean",
                            "description": "Whether the skill made real progress"
                        }
                    },
                    "required": ["is_progressive"],
                    "additionalProperties": False
                }
            }
        ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "skill_evaluate",
                    "description": "Analyze the skill and give me a evaluation on progressiveness",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_progressive": {
                                "type": "boolean",
                                "description": "Whether the skill made real progress"
                            },
                        },
                        "required": ["is_progressive"],
                        "additionalProperties": False
                    }
                }
            }       
        ]
    else:
        print(f"Model {model_name} not found")
        return None 
        
def cluster_skills_tool(model_name):
    if "claude" in model_name:
        return {
            "name": "cluster_skills",
            "description": (
                    "Cluster a list of new_skills semantically; return an array "
                    "called 'clusters', each with a representative name/description "
                    "and the list of member indices."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "clusters": {
                        "type": "array",
                        "description": "The list of clusters",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the skill"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "The description of the skill"
                                },
                                "members": {
                                    "type": "array",
                                    "description": "The list of member skills' id",
                                    "items": {"type": "integer"}
                                }
                            },
                            "required": ["name", "description", "members"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["clusters"],
                "additionalProperties": False
            }
        }
        
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return {
            "type": "function",
            "function": {
                "name": "cluster_skills",
                "description":  (
                    "Cluster a list of new_skills semantically; return an array "
                    "called 'clusters', each with a representative name/description "
                    "and the list of member indices."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "description": "The list of clusters",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "The name of the skill"
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "The description of the skill"
                                    },
                                    "members": {
                                        "type": "array",
                                        "description": "The list of member skills' indices",
                                        "items": {"type": "integer"}
                                    }
                                },
                                "required": ["name", "description", "members"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["clusters"],
                    "additionalProperties": False
                },
            }    
        }
    
    else:
        print(f"Model {model_name} not found")
        return None









def do_operation_tools(model_name):
    if "claude" in model_name:
        return [
             {
                "name": "Click",
                "description": "Click at the given coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "RightSingle",
                "description": "Right click at the given coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "LeftDouble",
                "description": "Double click at the given coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False
                },
            },
            {
                "name": "Type",
                "description": "Type the given text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"],
                    "additionalProperties": False
                },
            },
            {
                "name": "Drag",
                "description": "Drag from the first coordinates to the second coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"}
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                    "additionalProperties": False
                },
            },
            {
                "name": "Finished",
                "description": "Finish the task",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                },
            }
         ]
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return [
            {
                "type": "function",
                "function": {
                    "name": "Click",
                    "description": "Click at the given coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "RightSingle",
                    "description": "Right click at the given coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "LeftDouble",
                    "description": "Double click at the given coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"}
                        },
                        "required": ["x", "y"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Drag",
                    "description": "Drag from the first coordinates to the second coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                            "x2": {"type": "number"},
                            "y2": {"type": "number"}
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Finished",
                    "description": "Finish the task",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                }
            }
        ]
    else:
        print(f"Model {model_name} not found")
        return None
    
def merge_skills_tool(model_name):
    if "claude" in model_name:
        return {
            "name": "merge_skills",
            "description": (
                "Cluster raw new_skills among themselves and merge them into "
                "existing_skill_clusters by semantic similarity. "
                "Return an array 'clusters' where each item has:\n"
                "- id: reuse existing cluster's id, or for brand-new clusters use -1\n"
                "- name: representative name\n"
                "- description: representative description\n"
                "- members: list of integer skill IDs (from existing.members and/or new_skills[].id)"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":        {"type": "integer"},
                                "name":       {"type": "string"},
                                "description":{"type": "string"},
                                "members": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                }
                            },
                            "required": [
                                "id",
                                "name",
                                "description",
                                "members"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["clusters"],
                "additionalProperties": False
            }
        }
    elif model_name == "gpt-4o" or model_name == "o4-mini":
        return     {
            "type": "function",
            "function": {
                "name": "merge_skills",
                "description": (
                    "Cluster raw new_skills among themselves and merge them into "
                    "existing_skill_clusters by semantic similarity. "
                    "Return an array 'clusters' where each item has:\n"
                    "- id: reuse existing cluster's id, or for brand-new clusters use -1\n"
                    "- name: representative name\n"
                    "- description: representative description\n"
                    "- members: list of integer skill IDs (from existing.members and/or new_skills[].id)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id":        {"type": "integer"},
                                    "name":       {"type": "string"},
                                    "description":{"type": "string"},
                                    "members": {
                                        "type": "array",
                                        "items": {"type": "integer"}
                                    }
                                },
                                "required": [
                                    "id",
                                    "name",
                                    "description",
                                    "members"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["clusters"],
                    "additionalProperties": False
                }
            }
        }
    else:
        print(f"Model {model_name} not found")
        return None