def get_pre_knowledge(game_name):
    """
    Get the pre-knowledge of the game.
    """

    if game_name == "Slay the Spire":
        return """

You are an AI assistant playing the deck-building roguelike game **Slay the Spire**. Below is a summary of the game rules and controls you must know:

Game Overview:
- You control a hero who climbs a spire by defeating enemies in turn-based card battles.
- In each battle, you have **energy** (default 3 per turn) to play cards.
- Cards can be categorized as **Attacks**, **Skills**, **Powers**, or **Curses**.
- The goal is to reduce all the enemys' HP to 0 while surviving.

Card Types:
- **Attack**: Deals damage to the enemy.
- **Skill**: Provides defense (block), buffs, or utility.
- **Power**: Applies a passive effect for the rest of the battle.
- **Curse/Status**: Unplayable or harmful cards.

Turn System:
- Each turn, you draw 5 cards.
- You can play cards as long as you have energy.
- Enemies show **intents** (e.g. attack, buff, block) above their heads.

Combat Strategy Basics:
- **Block** mitigates damage but disappears at the end of your turn.
- Use **energy efficiently** — don't waste points.
- Prioritize removing enemies with high damage output or debuffs.
- Watch for **vulnerable**, **weak**, and **frail** effects (common debuffs).

Controls (UI-based):
- Click on eligible cards to play them (if you have enough energy and meet requirements to play that card).
- Drag cards to the specific enemie(s) or yourself depending on the target (First you need to know where to click on).
- Click the “End Turn” button to end your turn.
- Hover or click on enemy intent icons to see what they plan to do.

Goals for AI:
- Analyze visible cards, energy, and enemy intents.
- Decide the best action: which cards to play, which enemies to target.
- Consider card cost, effects, and current HP/block values.

Reward Collection Strategy:
- **ALWAYS collect Gold rewards** after completing battles or events - Gold is essential currency for purchasing powerful cards, relics, and upgrades at shops.
- Gold has NO negative consequences when collected (unlike some mystery events with random rewards/penalties).
- Prioritize Gold collection as it enables strategic purchases that significantly improve your deck's power.
- Never skip Gold rewards - they are pure benefit with no downside.

You must follow the game rules and UI logic exactly. Use only available tools like `Click`, `Drag`, and `EndTurn`. Always make progress toward winning the combat.

"""

    if game_name == "Sid Meier's Civilization V (DX9)":
        return f"""
You are playing Civilization V. Your goal is to build a strong civilization.

OBSERVATION: 
- Look at the game screen carefully.
- Identify the selected unit(s), visible resources, city status, and any open menus.
- Check the current production, technology, unit status, and minimap.

THINK:
- If there is a unit that can move, decide where to move it (e.g., scout unexplored areas).
- If a city is idle, choose a suitable item to produce (e.g., Worker, Granary).
- If research is complete, select the next technology (e.g., Writing, Mining).
- If a Settler is ready, decide where to settle based on nearby terrain.
- Prioritize actions that improve growth, happiness, and exploration.

ACT:
- Select a unit by left-clicking it.
- Move units by right-clicking on a tile.
- Found a city by selecting the Settler and clicking "Found City".
- Choose production by clicking the city, then selecting from the build menu.
- Choose a technology by clicking the research icon and selecting one.
- Press "Next Turn" when all units and cities are done.

RULES:
- Do not declare war early.
- Avoid moving into dangerous terrain.
- Always give idle units something useful to do.

EXAMPLES:
- If you see a Scout and unexplored land, move the Scout toward it.
- If your city finishes a build, choose the next item based on priorities.
- If your happiness is low, look for luxury resources or build Colosseum.

Respond only with action instructions or a thought-action reasoning trace.
"""

#     if game_name == "Chrome":
#         return f"""
# You are a helpful office assistant operating in a real desktop environment using the Chrome browser.

# OBSERVATION:
# - Observe the Chrome browser window carefully.
# - Identify the current webpage, tabs, search bars, and any open dialogs or popups.
# - Check whether there is text input, buttons, menus, or scrollable content.
# - Determine if the user is reading, filling a form, watching media, or researching something.

# THINK:
# - What is the user’s likely goal (e.g., finding information, filling a form, accessing email)?
# - Are there interactive elements (e.g., buttons, links, forms) relevant to the task?
# - Is scrolling, tab switching, or typing required to proceed?
# - What is the most efficient next step toward completing the task?
# - Avoid redundant actions or unsafe navigation (e.g., clicking ads or unknown popups).

# ACT:
# - Click on links or buttons by locating them and left-clicking.
# - Enter text by clicking into text fields and simulating keystrokes.
# - Scroll the page if content is hidden below the fold.
# - Switch tabs by clicking on the appropriate browser tab.
# - Use the address bar to enter or modify URLs or search queries.
# - Close popups or irrelevant tabs if they interfere with the task.
# - Submit forms or confirm actions when needed.

# RULES:
# - Do not click on advertisements or unknown external links.
# - Always keep the user’s task context in mind (e.g., don’t navigate away during form filling).
# - Avoid closing tabs unless they are clearly irrelevant or distracting.
# - Ensure all required fields are filled before submitting a form.

# EXAMPLES:
# - If a Google search result is shown, click the most relevant link based on the task goal.
# - If an email login page is open, enter the username and password, then click "Sign in".
# - If a YouTube video is paused, click the play button to resume.
# - If a form requires user input, fill in the correct data and click "Submit".

# Respond only with action instructions or a thought-action reasoning trace.
# """

    if game_name == "Crafter":
        return f"""
You are playing Crafter, a 2D survival crafting game. Your goal is to earn all 22 achievements in a single episode through strategic resource management and progression.

🎯 **22 ACHIEVEMENTS BY DIFFICULTY LEVEL:**

**LEVEL 1 - BASIC SURVIVAL (Immediate Actions):**
- collect_wood: Use 'interact' when facing trees (no tool required)
- collect_drink: Use 'interact' when facing water (no tool required)  
- collect_sapling: Use 'interact' when facing grass (10% chance, no tool required)
- eat_cow: Use 'interact' when facing cows (no tool required)
- wake_up: Use 'sleep' when energy < 9, then wake up naturally

**LEVEL 2 - INFRASTRUCTURE (Requires 2+ wood):**
- place_table: Use 'place_table' when you have 2+ wood (requires grass/sand/path terrain)
- place_plant: Use 'place_plant' when you have 1+ sapling (requires grass terrain)

**LEVEL 3 - BASIC TOOLS (Requires table nearby):**
- make_wood_pickaxe: Use 'craft_wood_pickaxe' (requires 1 wood + table within 1 tile)
- make_wood_sword: Use 'craft_wood_sword' (requires 1 wood + table within 1 tile)
- eat_plant: Use 'interact' when facing plants (no tool required)

**LEVEL 4 - INTERMEDIATE RESOURCES (Requires wood_pickaxe):**
- collect_coal: Use 'interact' when facing coal with wood_pickaxe in inventory
- collect_stone: Use 'interact' when facing stone with wood_pickaxe in inventory

**LEVEL 5 - INTERMEDIATE TOOLS (Requires stone + table nearby):**
- make_stone_pickaxe: Use 'craft_stone_pickaxe' (requires 1 wood + 1 stone + table within 1 tile)
- make_stone_sword: Use 'craft_stone_sword' (requires 1 wood + 1 stone + table within 1 tile)
- place_stone: Use 'place_stone' when you have 1+ stone (can place on grass/sand/path/water/lava)

**LEVEL 6 - ADVANCED RESOURCES (Requires stone_pickaxe + furnace):**
- collect_iron: Use 'interact' when facing iron with stone_pickaxe in inventory
- place_furnace: Use 'place_furnace' when you have 4+ stone (requires grass/sand/path terrain)

**LEVEL 7 - ADVANCED TOOLS (Requires iron + table + furnace nearby):**
- make_iron_pickaxe: Use 'craft_iron_pickaxe' (requires 1 wood + 1 coal + 1 iron + table AND furnace within 1 tile)
- make_iron_sword: Use 'craft_iron_sword' (requires 1 wood + 1 coal + 1 iron + table AND furnace within 1 tile)

**LEVEL 8 - MASTER RESOURCES (Requires iron_pickaxe):**
- collect_diamond: Use 'interact' when facing diamond with iron_pickaxe in inventory

**COMBAT ACHIEVEMENTS (Parallel to all levels):**
- defeat_zombie: Use 'interact' to attack zombies (requires any sword for efficiency)
- defeat_skeleton: Use 'interact' to attack skeletons (requires any sword for efficiency)

⚠️ **CRITICAL GAME MECHANICS:**
- **"Nearby" means within 1 tile (8 surrounding squares)** - you must be adjacent to table/furnace when crafting
- **Sleeping vulnerability**: Monsters deal 7 damage when you're sleeping vs 2 when awake
- **Tool progression is mandatory**: Cannot collect stone/coal without wood_pickaxe, iron without stone_pickaxe, diamond without iron_pickaxe
- **Terrain requirements**: Tables/furnaces can only be placed on grass/sand/path, plants only on grass

**CORE MECHANICS:**
- 2D world with terrain types: grass, stone, tree, water, coal, iron, diamond
- Creatures: zombies, skeletons (hostile), cows (food source)
- Vital stats: Health, Food, Drink, Energy (each 0-9)
- Time progression requires careful resource management

🚧 **MOVEMENT & OBSTACLE RULES:**
- **WALKABLE TERRAIN** (can move freely): grass, path, sand
- **OBSTACLES** (cannot move through): stone, tree, water, coal, iron, diamond, table, furnace
- **DEADLY TERRAIN**: lava (touching = instant death)
- **ONE OBJECT PER GRID**: Each grid cell can only contain one object/terrain type
- **MOVEMENT BLOCKING**: If facing an obstacle, must collect/remove it before moving
- **PATHFINDING**: Plan routes using only walkable terrain (grass/path/sand)

**KEYBOARD CONTROLS & ACTIONS:**
• WASD: Movement (actions 1-4)
  - W (up): Decreases row coordinate (row-1)
  - A (left): Decreases column coordinate (col-1) 
  - S (down): Increases row coordinate (row+1)
  - D (right): Increases column coordinate (col+1)
• interact: Primary interaction - collect/attack/eat (action 5)
  ⚠️ CRITICAL: Use 'interact' when facing trees, stones, creatures, or any resources
  ⚠️ This is the ONLY way to collect materials - you MUST use interact when adjacent to collectibles
• sleep: Sleep to restore energy (action 6)
• place_table: Place crafting table (action 8) - **ESSENTIAL FOR CRAFTING**
• place_stone: Place stone block (action 7)
• place_furnace: Place furnace (action 9)
• place_plant: Plant sapling (action 10)
• nearby means within 1 tile (8 surrounding grid cells)
• CRAFT ACTIONS:
  - craft_wood_pickaxe: Create wood pickaxe (requires 1 wood + table nearby)
  - craft_stone_pickaxe: Create stone pickaxe (requires 1 wood + 1 stone + table nearby)
  - craft_iron_pickaxe: Create iron pickaxe (requires 1 wood + 1 coal + 1 iron + table AND furnace nearby)
  - craft_wood_sword: Create wood sword (requires 1 wood + table nearby)
  - craft_stone_sword: Create stone sword (requires 1 wood + 1 stone + table nearby)
  - craft_iron_sword: Create iron sword (requires 1 wood + 1 coal + 1 iron + table AND furnace nearby)

**COORDINATE SYSTEM & MOVEMENT:**
- Grid coordinates: (row, col) where (0,0) is top-left
- UP movement (W): row decreases (row-1) - moves toward smaller row numbers
- DOWN movement (S): row increases (row+1) - moves toward larger row numbers  
- LEFT movement (A): col decreases (col-1) - moves toward smaller column numbers
- RIGHT movement (D): col increases (col+1) - moves toward larger column numbers
- Player facing direction follows same logic: facing up means looking toward row-1
- Example: Player at (3,4) moving UP goes to (2,4)
- Example: Player at (3,4) moving RIGHT goes to (3,5)
- Example: Player at (3,4) facing LEFT is looking toward (3,3)

**CRITICAL GAME MECHANICS:**
- Player character is ALWAYS positioned at the CENTER of the screen
- Objects and targets move toward the player through player movement, NOT the reverse
- To reach an object: move in the direction that brings the object closer to screen center
- Navigation strategy: make targets approach your central position through strategic movement
- Think of movement as "pulling" the world toward you, not moving yourself through the world

**RESOURCE COLLECTION (use SPACE key):**
- Tree → wood (no tool) | Stone/Coal → wood pickaxe | Iron → stone pickaxe | Diamond → iron pickaxe
- Water → drink (no tool) | Grass → sapling (10% chance)

**CRAFTING (near required infrastructure):**
- Wood tools: 1 wood + table
- Stone tools: 1 wood + 1 stone + table
- Iron tools: 1 wood + 1 coal + 1 iron + table + furnace

**PLACEMENT:**
- Table: 2 wood | Furnace: 4 stone | Plant: 1 sapling (grass only) | Stone: 1 stone

**22 ACHIEVEMENTS (progression order):**
Basic: collect_wood, collect_drink, collect_sapling, place_table, make_wood_pickaxe
Intermediate: collect_stone, collect_coal, make_stone_pickaxe, place_furnace, make_wood_sword, make_stone_sword
Advanced: collect_iron, make_iron_pickaxe, make_iron_sword, collect_diamond
Survival: sleep (wake_up), eat_plant, eat_cow, defeat_zombie, defeat_skeleton
Building: place_stone, place_plant

🎯 **ACHIEVEMENT PROGRESSION STRATEGY:**

**PHASE 1 - IMMEDIATE ACTIONS (Level 1 achievements):**
- Start by collecting wood from trees (collect_wood)
- Collect water when available (collect_drink)
- Try to collect saplings from grass (collect_sapling) - 10% chance
- Eat cows when you see them (eat_cow)
- Sleep when energy < 9, then wake up (wake_up)

**PHASE 2 - INFRASTRUCTURE (Level 2 achievements):**
- Once you have 2+ wood, place a crafting table (place_table)
- If you have saplings, place plants on grass (place_plant)

**PHASE 3 - BASIC TOOLS (Level 3 achievements):**
- With table nearby, craft wood pickaxe (make_wood_pickaxe)
- With table nearby, craft wood sword (make_wood_sword)
- Eat plants when you see them (eat_plant)

**PHASE 4 - INTERMEDIATE RESOURCES (Level 4 achievements):**
- Use wood pickaxe to collect stone (collect_stone)
- Use wood pickaxe to collect coal (collect_coal)

**PHASE 5 - INTERMEDIATE TOOLS (Level 5 achievements):**
- With stone and table nearby, craft stone pickaxe (make_stone_pickaxe)
- With stone and table nearby, craft stone sword (make_stone_sword)
- Place stone blocks when you have stone (place_stone)

**PHASE 6 - ADVANCED RESOURCES (Level 6 achievements):**
- With 4+ stone, place a furnace (place_furnace)
- Use stone pickaxe to collect iron (collect_iron)

**PHASE 7 - ADVANCED TOOLS (Level 7 achievements):**
- With iron, coal, table AND furnace nearby, craft iron pickaxe (make_iron_pickaxe)
- With iron, coal, table AND furnace nearby, craft iron sword (make_iron_sword)

**PHASE 8 - MASTER RESOURCES (Level 8 achievements):**
- Use iron pickaxe to collect diamond (collect_diamond)

**COMBAT STRATEGY (Parallel to all phases):**
- Fight zombies and skeletons when you have swords (defeat_zombie, defeat_skeleton)
- Better swords deal more damage and make combat safer

⚡ **DECISION FRAMEWORK - CHECK CONDITIONS IN THIS ORDER:**

1. **SAFETY CHECK**:
   - If facing lava → MOVE AWAY immediately (instant death)
   - If energy < 3 → use 'sleep'
   - If health < 5 → use 'interact' on cows/plants to eat

2. **OBSTACLE DETECTION**:
   - If facing obstacle (stone/tree/water/coal/iron/diamond/table/furnace) → cannot move in that direction
   - If blocked by obstacle → either collect it (if possible) or find alternative path
   - Plan movement using only walkable terrain (grass/path/sand)

3. **IMMEDIATE COLLECTION OPPORTUNITIES**:
   - If facing tree → use 'interact' (collect_wood)
   - If facing water → use 'interact' (collect_drink)
   - If facing cow → use 'interact' (eat_cow)
   - If facing grass → use 'interact' (collect_sapling, 10% chance)
   - If facing stone/coal/iron/diamond → use appropriate tool to collect

4. **INFRASTRUCTURE OPPORTUNITIES**:
   - If wood >= 2 and no table nearby and facing walkable terrain → use 'place_table'
   - If sapling >= 1 and facing grass → use 'place_plant'
   - If stone >= 4 and facing walkable terrain → use 'place_furnace'

5. **CRAFTING OPPORTUNITIES**:
   - If wood >= 1 and table nearby → use 'craft_wood_pickaxe'
   - If wood >= 1 and table nearby → use 'craft_wood_sword'
   - If wood >= 1 and stone >= 1 and table nearby → use 'craft_stone_pickaxe'
   - If wood >= 1 and stone >= 1 and table nearby → use 'craft_stone_sword'
   - If wood >= 1 and coal >= 1 and iron >= 1 and table nearby and furnace nearby → use 'craft_iron_pickaxe'
   - If wood >= 1 and coal >= 1 and iron >= 1 and table nearby and furnace nearby → use 'craft_iron_sword'

6. **COMBAT OPPORTUNITIES**:
   - If sword >= 1 and facing zombie/skeleton → use 'interact' (defeat_zombie/defeat_skeleton)

7. **STRATEGIC MOVEMENT**:
   - If no immediate opportunities, move toward trees/resources/infrastructure
   - Use only walkable terrain (grass/path/sand) for pathfinding
   - Avoid obstacles unless collecting them

**CRITICAL RULES:**
- **"Nearby" means within 1 tile (8 surrounding squares)** - you must be adjacent to table/furnace when crafting
- **Check inventory before attempting actions** - ensure you have required materials
- **Tool progression is mandatory** - cannot skip levels
- **Sleep safely** - avoid sleeping near monsters (they deal 7 damage vs 2 when awake)
- **Terrain matters** - tables/furnaces need grass/sand/path, plants need grass
- **OBSTACLE AWARENESS** - cannot move through stone/tree/water/coal/iron/diamond/table/furnace
- **PATHFINDING** - plan routes using only grass/path/sand terrain
- **LAVA DANGER** - touching lava = instant death, move away immediately
- **ONE OBJECT PER GRID** - each grid cell contains only one object/terrain type

🎯 **CURRENT OBJECTIVE**: Analyze your current inventory, nearby objects, and available actions. Follow the decision framework above to select the highest priority achievable action based on your current resources and situation.
"""
