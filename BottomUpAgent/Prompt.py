import json

def generate_skill_prompt(operations):
    return f"""You are playing a game.

Objectives:
- Analyze the recent UI-level operations you performed.
- Determine if the operation sequence is complete or requires continuation.

Inputs:
1. A list of UI-level operations: '{operations}'
2. Screenshots of the game captured during these operations.

Instructions:
1. Determine which UI elements were interacted with, based on the coordinates provided.
2. Analyze how the game screen changed during the operations.
3. **Assess Operation Completeness:**
   - COMPLETE: The operation achieves a self-contained, meaningful action (e.g., "Open menu", "Close dialog", "Complete card play")
   - INCOMPLETE: The operation starts an action but requires follow-up steps (e.g., "Select card" but target not chosen, "Open submenu" but no selection made)
4. Give this sequence of operations a meaningful **name** and a **description**.
5. **Decision Logic:**
   - If operations are **meaningless** → use `no_meaning_skill` tool
   - If operations are **complete and meaningful** → use `save_skill` tool
   - If operations are **incomplete but meaningful** → use `incomplete_skill` tool with next_action_hint
6. For incomplete operations, provide a clear hint about what should happen next.
7. Think step by step before making a decision.

**Examples:**
- Clicking a card in hand → INCOMPLETE (needs target selection)
- Clicking a card then clicking enemy → COMPLETE (full attack sequence)
- Opening a menu → INCOMPLETE (needs menu selection)
- Opening menu then selecting option → COMPLETE (full menu interaction)
"""
            

def get_skill_guidance_prompt(task, potential_interact_points, existed_actions):
    return f"""You are a game teacher. I am learning to perform the task: '{task}'.

Context:
- I will provide you with the current game screenshot.
- You will also receive a list of potential interaction points: {potential_interact_points}
- I have already learned and performed some actions: {existed_actions}

Your Role:
- Help me discover new and useful operations.
- Avoid repeating actions I've already learned.
- Guide me toward the most effective interaction, based on the current game state.

Instructions:
1. Analyze the screenshot to understand the current game state.
2. Choose the most appropriate interaction point from the list of potential interact points.
3. Your guidance should **not** duplicate any of the existing actions.
4. Use the tool `give_guidance` to provide your recommendation.
5. Think step by step before making your decision.
"""



def select_skill_prompt(skills):
    return f"""You are playing a game.

Objectives:
- Select the best skill from the provided skill list.

Inputs:
1. A list of learned skills.
2. A screenshot of the current game state.

Skills:
{skills}

Instructions:
1. Analyze the screenshot to understand the current state of the game.
2. For each skill, assess:
   - Whether the **execution conditions are currently satisfied**.
   - Whether it is functionally suitable for current state.
3. **First, eliminate any skills that are not currently executable.**
4. Then, among the remaining executable skills, choose the one most appropriate.
5. Must return your selection using the function tool provided — do NOT respond with text.
6. Think step by step before making your final decision.
"""

def skill_evaluate_prompt(task, skill_info):
    return f"""You are an assistant whose sole responsibility is to analyze a single in-game action against a task, and return **exactly one** `function_call` to `action_reflex`.

Task:
- '{task}'

Inputs:
1. **Action Information**  
   Name and description of the action:  
   {skill_info}

2. **Screenshots**  
   A pair of images showing the game state **before** and **after** the action.

Instructions:

Step 1: **Expected Change**  
- Based on the action description, summarize what is expected to change in the game state if the action works as intended.

Step 2: **Actual Change**  
- Analyze the differences between the before and after screenshots.
- Describe all **observable UI changes** (e.g., elements, values, progress bars, icons).

Step 3: **Consistency Check → `is_consistent`**  
- If the actual changes match the expected outcome (in type and magnitude), set `is_consistent = true`; otherwise, `false`.

Step 4: **Progress Indicators**  
- Identify at least **three concrete, task-relevant indicators** that suggest progress toward the task goal.

Step 5: **Progress Check → `is_progressive`**  
- Evaluate whether any of the indicators were clearly met.
  - If **any** indicator is met meaningfully, set `is_progressive = true`.
  - If **none** are met, or if changes are negligible/irrelevant, set `is_progressive = false`.
  - If the action is **only preparatory** (e.g., hovering, previewing, selecting without effect), set `is_progressive = false`.

Step 6: **Default to False**  
- If information is incomplete, unclear, or ambiguous, default both booleans to `false`.

Remember:
→ Do not guess.
→ Only return the `action_reflex` function call with the evaluated booleans.
"""

def skill_evaluate2_prompt(task):
    return f"""You are an assistant whose sole responsibility is to analyze a single in-game action against a task, and return **exactly one** `function_call` to `action_reflex`.

Task:
- '{task}'

Inputs:
1. **Screenshots**  
   A pair of images showing the game state **before** and **after** the action.

Instructions:

Step 1: **Progress Indicators**  
- Identify at least **three concrete, task-relevant indicators** that suggest progress toward the task goal.

Step 2: **Progress Check → `is_progressive`**  
- Evaluate whether any of the indicators were clearly met.
  - If **any** indicator is met meaningfully, set `is_progressive = true`.
  - If **none** are met, or if changes are negligible/irrelevant, set `is_progressive = false`.
  - If the action is **only preparatory** (e.g., hovering, previewing, selecting without effect), set `is_progressive = false`.

Step 6: **Default to False**  
- If information is incomplete, unclear, or ambiguous, default to `false`.

Remember:
→ Do not guess.
→ Only return the `action_reflex` function call with the evaluated booleans.
"""

def do_operation_prompt(task):
    return f"""You are playing a game and your task is: '{task}'.

Objective:
- Determine the next specific action to perform based on the current screen image and the given task.

Inputs:
1. The current screenshot of the game.

Instructions:
1. Carefully analyze the current image to understand the game state.
2. Select the most appropriate action to make progress on the task.
3. The action must be selected from the tools I provide.
4. Any coordinates used must align with the actual image resolution.
5. Only return the action decision using the function tool provided.

Think step by step before making your final decision."""

def do_operation_prompt_v2(task):
    return f"""You are playing a game and your task is: '{task}'.

Determine the next specific action to perform based on the current screen image and the given task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x2,y2)<|box_end|>')
                                         
## Inputs
1. The current screenshot of the game.

## User Instruction
1. Carefully analyze the current image to understand the game state.
2. Select the most appropriate action from the Action Space to make progress on the task.
3. Any coordinates used must align with the actual image resolution. Output the resolution of the image you received.
4. Only return the action decision using one of the Action Space defined above.

Think step by step before making your final decision."""
        
def cluster_skills_prompt(skills):
    return f"""

Background:  
You are an assistant for grouping similar skills.  
Input: a JSON array called “new_skills”.  

Task: Identify skills that are **nearly identical in meaning or function**, even if their expressions differ.  
• Group together skills that **essentially perform the same task or behavior**, even if worded differently.  
• Do **not** group skills that express **different functions or intentions**, even if they appear related.  
• Think of this as merging duplicates or near-duplicates, not broad semantic clustering.  

Output must strictly call the function “cluster_skills” with no extra text.

Here is the list of new_skills (id, name, description):  
{json.dumps(skills, indent=2)}

Please:  
1. Identify cluster of **functionally equivalent** skills.  
2. For each cluster, select a representative “action_name” and “action_description”.  
3. List its members as an array of action ids.  

"""

def merge_skills_prompt(existing_skill_clusters, new_skills):
    return f"""
Background:
You are an assistant that merges and clusters skills in one call.
Input:
    • existing_skill_clusters: clusters with cluster_id, name, description, members
    • new_skills: raw skills with id, name, description
Instruction:
    1) Cluster new_skills among themselves by semantic similarity.
    2) For each resulting new cluster:
        - If it matches an existing cluster, merge into it (reuse that cluster_id).
        - Otherwise, assign cluster_id = -1.
    3) Each output cluster must include:
        - cluster_id
        - name & description (representative)
        - members: combined list of all action IDs.
Output:
    Exactly one function_call to "merge_skills", no extra text.

existing_skill_clusters:
{json.dumps(existing_skill_clusters, indent=2)}

new_skills:
{json.dumps(new_skills, indent=2)}

Please perform the merge as specified.
Return the merged list under the key "clusters".

"""


def parse_response(response):
    """
    Parse the response from the model to extract the function call and its arguments.
    """
    try:
        # Attempt to parse the response as JSON
        parsed_response = json.loads(response)
        if 'function_call' in parsed_response:
            return parsed_response['function_call']
        else:
            raise ValueError("No function call found in the response.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the response.")