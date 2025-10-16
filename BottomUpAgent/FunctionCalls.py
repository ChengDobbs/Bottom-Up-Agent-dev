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
                            "type": "object",
                            "description": "Structured hint containing specific action details for continuation",
                            "properties": {
                                "action_type": {
                                    "type": "string",
                                    "description": "Type of action to perform",
                                    "enum": ["Click", "Drag", "Key"]
                                },
                                "target_coordinates": {
                                    "type": "array",
                                    "description": "Exact [x, y] coordinates or [x1, y1, x2, y2] coordinates for the action",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 4
                                },
                                "target_object_id": {
                                    "type": "string",
                                    "description": "ID of the target object (if applicable)"
                                },
                                "keyboard_key": {
                                    "type": "string",
                                    "description": "Specific keyboard key to press (if action_type is 'Key')"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of why this specific action should be taken next"
                                },
                                "expected_outcome": {
                                    "type": "string",
                                    "description": "What should happen after performing this action"
                                }
                            },
                            "required": ["action_type", "reasoning", "expected_outcome"],
                            "additionalProperties": False
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