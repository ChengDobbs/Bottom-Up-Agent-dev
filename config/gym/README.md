# Gym Environment Configurations

This directory contains various Gym environment configuration files for the Bottom-Up Agent framework. All environments feature **discrete action spaces** and **interactive gameplay**, suitable for GUI operation scenarios that require user response.

## 📁 Configuration Files Overview

### 1. Crafter Environment (`crafter_config.yaml`) ⭐
- **Environment Type**: Open-world survival and crafting game
- **Action Space**: 17 discrete actions (movement + interaction + crafting)
- **Observation Space**: RGB images (64x64x3)
- **Features**: Complex long-term exploration with 22 achievement system
- **Use Cases**: Unsupervised exploration, long-term planning, skill discovery
- **Special Settings**: 
  - GUI configuration with input timeout and keyboard mapping
  - Achievement tracking and success rate computation
  - Interactive mode with human teacher support

### 2. Blackjack Environment (`blackjack_config.yaml`)
- **Environment Type**: Classic 21-point card game
- **Action Space**: 2 discrete actions (Hit/Stick)
- **Observation Space**: Tuple (player sum, dealer card, usable ace)
- **Features**: Probabilistic decision making and risk assessment
- **Use Cases**: Decision theory testing, probability reasoning
- **Special Settings**:
  - Natural blackjack bonus enabled
  - Human interaction for strategic decisions
  - Extended execution duration (1.5s) for card game decisions

### 3. FrozenLake Environment (`frozenlake_config.yaml`)
- **Environment Type**: Grid world navigation
- **Action Space**: 4 discrete actions (Left/Down/Right/Up)
- **Observation Space**: Discrete states (16 positions in 4x4 grid)
- **Features**: Simple grid navigation, avoiding holes to reach goal
- **Use Cases**: Basic navigation testing, strategy learning introduction
- **Special Settings**:
  - Non-slippery mode for deterministic actions
  - 4x4 grid map configuration
  - Human interaction for discrete decisions

### 4. Taxi Environment (`taxi_config.yaml`)
- **Environment Type**: Taxi dispatch and passenger service game
- **Action Space**: 6 discrete actions (South/North/East/West/Pickup/Dropoff)
- **Observation Space**: Discrete states (500 total states)
- **Features**: Complex task planning, passenger pickup and delivery
- **Use Cases**: Multi-step task planning, reward engineering testing
- **Special Settings**:
  - Deterministic movement (no rain)
  - Fixed passenger destinations
  - Extended episode length (200 steps)

### 5. Simple Gym Environment (`gym_simple_config.yaml`)
- **Environment Type**: Basic Gym environment wrapper (default: CartPole-v1)
- **Action Space**: Configurable (CartPole: 2 discrete actions)
- **Observation Space**: Configurable (CartPole: Box[4] continuous)
- **Features**: Lightweight configuration for standard Gym environments
- **Use Cases**: Quick testing, basic agent development, standard benchmarks
- **Special Settings**:
  - Interactive mode toggle
  - GUI visualization options
  - Training and evaluation configurations

### 6. Simple GUI Test (`simple_gui_test.yaml`)
- **Environment Type**: Minimal test configuration (CartPole-v1)
- **Action Space**: 2 discrete actions (Left/Right)
- **Observation Space**: Box[4] (cart position, velocity, pole angle, pole velocity)
- **Features**: Lightweight testing with disabled heavy components
- **Use Cases**: GUI testing, quick validation, minimal resource usage
- **Special Settings**:
  - Disabled exploration, reflexion, and reset mechanisms
  - Simple random policy
  - Disabled SOM, YOLO, OmniParser, and CLIP components

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Basic dependencies
pip install gymnasium

# Crafter environment (if using crafter_config.yaml)
pip install crafter
pip install pygame  # For human-computer interaction interface

# Additional dependencies for specific environments
pip install numpy pillow pyyaml
```

### 2. Test Configurations
```bash
# Test all configurations
python -m demos.demo_gym_configs

# Test Crafter environment specifically
python -m demos.crafter_interactive_launcher --resolution 400 --max-steps 1_000_000

# Test simple environments
python -m demos.demo_simple_gym
```

### 3. Choose Configuration File
Select the appropriate configuration file based on your needs:
- **Beginners**: `simple_gui_test.yaml` or `frozenlake_config.yaml` - Simple and easy to understand
- **Intermediate Users**: `taxi_config.yaml` or `blackjack_config.yaml` - Moderate complexity
- **Advanced Users**: `crafter_config.yaml` - Complex open-world environment
- **Quick Testing**: `gym_simple_config.yaml` - Standard Gym environments

## 🔧 Configuration File Structure

Each configuration file contains the following main sections:

```yaml
# Project basic information
project_name: 'GymAgent'
game_name: 'EnvironmentName'
run_name: 'description'
close_explore: False
close_reflexion: False
close_reset: False
is_base: False
use_mcp: True

# MCP (Monte Carlo Planning) configuration
mcp:
  max_iter: 8

# Execution settings
result_path: 'results'
exec_duration: 1.0  # Wait time for user interaction
teacher_type: 'Human'  # Human interaction support

# Brain/AI model configuration
brain:
  base_model: 'claude-3-7-sonnet-20250219'
  evaluate_model: 'gpt-4o'
  uct_c: 5.0
  uct_threshold: 0.0

# Environment configuration
gym_env:  # or gym:
  env_id: 'EnvironmentID'
  render_mode: 'human'  # or 'rgb_array'
  max_episode_steps: 1000
  
  # Action and observation spaces
  action_space:
    type: 'Discrete'
    n: 4  # Number of actions
  
  observation_space:
    type: 'Box'  # or 'Discrete'
    shape: [64, 64, 3]  # If image-based

# Memory configurations
long_memory:
  sim_threshold: 0.85

vector_memory:
  similarity_threshold: 0.92
  embedding_type: "open_clip"
  clip_model: "ViT-B/32"
  vector_dim: 512
```

## 🎮 Environment Comparison

| Environment | Complexity | Observation Type | Actions | Episode Length | Use Cases |
|-------------|------------|------------------|---------|----------------|-----------|
| Simple GUI Test | ⭐ | Continuous | 2 | Short | Quick testing |
| FrozenLake | ⭐ | Discrete state | 4 | Short | Basic navigation |
| Blackjack | ⭐⭐ | Tuple | 2 | Short | Decision theory |
| Taxi | ⭐⭐⭐ | Discrete state | 6 | Medium | Task planning |
| Simple Gym | ⭐⭐⭐ | Configurable | Variable | Variable | Standard benchmarks |
| Crafter | ⭐⭐⭐⭐⭐ | RGB images | 17 | Long | Open-world exploration |

## 🔍 Usage Recommendations

### For GUI Operation Agents
1. **Simple GUI Test**: Perfect for basic GUI component testing
2. **FrozenLake**: Suitable for testing basic directional key operations
3. **Taxi**: Good for testing complex key sequences
4. **Blackjack**: Ideal for testing simple binary decision making
5. **Crafter**: Best for testing complex multimodal interactions

### For Reinforcement Learning Agents
1. Start with simple environments (Simple GUI Test, FrozenLake)
2. Gradually increase complexity (Taxi → Blackjack)
3. Challenge with standard benchmarks (Simple Gym)
4. Ultimate challenge with open-world (Crafter)

### For Interactive Development
- All configurations support human teacher interaction
- Extended execution durations allow for thoughtful decision making
- Visual rendering modes provide immediate feedback
- MCP integration enables sophisticated planning

## 🛠️ Custom Configuration

You can create your own configuration files based on existing ones:

1. Copy a similar configuration file
2. Modify environment ID and parameters
3. Adjust MCP and memory settings
4. Configure action/observation spaces
5. Test with `demo_gym_configs.py`

### Example Custom Configuration
```yaml
project_name: 'MyCustomAgent'
game_name: 'LunarLander-v2'
run_name: 'lunar-lander-test'

gym_env:
  env_id: 'LunarLander-v2'
  render_mode: 'human'
  max_episode_steps: 1000
  
  action_space:
    type: 'Discrete'
    n: 4  # Do nothing, fire left, fire main, fire right
    
  observation_space:
    type: 'Box'
    shape: [8]  # Position, velocity, angle, etc.
    dtype: 'float32'
```

## 📊 Performance Recommendations

- **Simple GUI Test/FrozenLake/Blackjack**: Lightweight, suitable for rapid iteration
- **Taxi**: Moderate computational requirements
- **Simple Gym**: Varies by chosen environment
- **Crafter**: Computationally intensive, GPU acceleration recommended
- For long-term training, disable rendering (`render_mode: null`)
- Use vectorized environments to improve training efficiency
- Adjust `exec_duration` based on decision complexity

## 🐛 Common Issues

### Q: Environment creation failed?
A: Ensure corresponding dependency packages are installed, especially Crafter requires additional installation.

### Q: Rendering issues?
A: When running on headless servers, set `render_mode: 'rgb_array'` or `null`.

### Q: Crafter environment not found?
A: Crafter uses direct creation method (`crafter.Env()`), not through `gym.make()`.

### Q: How to add new environments?
A: Reference existing configuration file structures, add new YAML files and test them.

### Q: MCP integration issues?
A: Ensure `use_mcp: True` and proper `brain` model configuration.

### Q: Memory configuration errors?
A: Check `long_memory` and `vector_memory` settings, ensure CLIP models are available.

## 🔧 Advanced Features

### Human Teacher Integration
- All configurations support `teacher_type: 'Human'`
- Interactive decision making with extended execution durations
- Real-time feedback and guidance capabilities

### Memory Systems
- Long-term memory with similarity thresholds
- Vector databases with CLIP embeddings
- Configurable memory retention and retrieval

### Planning Integration
- Monte Carlo Planning (MCP) with configurable iterations
- UCT (Upper Confidence bounds applied to Trees) parameters
- Exploration vs exploitation balance

## 📚 Related Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Crafter Environment](https://github.com/danijar/crafter)
- [Bottom-Up Agent Framework](../README.md)
- [OpenAI Gym Environments](https://gym.openai.com/envs/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## 🎯 Environment-Specific Notes

### Crafter Environment
- Requires pygame for interactive GUI
- Supports 22 different achievements
- Complex crafting and survival mechanics
- Long-term exploration and planning required

### Blackjack Environment
- Implements standard blackjack rules
- Natural blackjack bonus available
- Probabilistic outcomes require strategic thinking
- Short episodes with immediate feedback

### FrozenLake Environment
- Deterministic movement (non-slippery)
- Simple 4x4 grid world
- Clear goal-oriented navigation
- Excellent for learning basic RL concepts

### Taxi Environment
- Complex state space (500 states)
- Multi-objective tasks (pickup + dropoff)
- Hierarchical action planning required
- Good for testing planning algorithms

---

**Tip**: Start testing with simple environments to ensure proper framework integration before attempting complex environments. Each configuration is designed to work seamlessly with the Bottom-Up Agent architecture while providing unique challenges and learning opportunities.