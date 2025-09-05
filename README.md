
# BottomUpAgent
## Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution



<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.17673-green.svg)](https://arxiv.org/abs/2505.17673)  [![Python Version](https://img.shields.io/badge/Python-3.10-yellow.svg)]()  [![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)]()


</div>

## Introduction

This repository contains the official implementation of the paper  
**“Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution”**.  
Our bottom-up agents learn skills through autonomous exploration and reasoning—starting from raw pixel inputs and simulated mouse/keyboard actions, evolving competence purely from experience.

![Intro](figs/intro.jpg )

## Project Highlights

- **Zero-prior Learning**: Agents operate without predefined goals, APIs, or game knowledge.  
- **Unified Codebase**: One framework supports multiple environments (e.g., Slay the Spire, Civilization V).  
- **Experience-Driven Evolution**: Skills are discovered, refined, and shared dynamically across agents.  
- **Visualization**: Execution states and skill libraries can be visualized via integrated GUIs and logs.

## Demo

| Environment                                                                 | Demo GIF                                           |
|:---------------------------------------------------------------------------:|:--------------------------------------------------:|
| [Slay the Spire](https://www.megacrit.com/games/#slay-the-spire)           | <img src="figs/sts_skill_augment_x5.gif" alt="Slay the Spire Demo" width="480"/> |
| [Civilization V](https://civilization.2k.com/civ-v/)                       | <img src="figs/c5_skill_reuse_x5.gif"           alt="Civ V Demo"          width="480"/> |

# Table of Contents

- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Citation](#citation)  
- [License](#license)  

# Installation
> 💡 Currently, our implementation supports both **Windows** and **Linux**.  
We select games based on the principle of **cross-platform availability via Steam**, ensuring support across **Windows**, **macOS**, and **SteamOS/Linux**, and will improve the adaptation for all platforms in the future.

## 1. Clone the repository 
```bash
git clone https://github.com/AngusDujw/Bottom-Up-Agent.git
cd Bottom-Up-Agent
```


## 2. Create & activate a Conda environment 
```bash
conda create -n bottomup python=3.10 -y
conda activate bottomup
```
## 3. Install dependencies
### 3.1 Install PaddlePaddle
Please find your OS and CUDA version in [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/en/install/quick). The current implementation relies on PaddlePaddle 3.1.0 with CUDA 11.8.
```bash
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
For CPU users:

```bash
pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

You can check the PaddlePaddle installation by running:
```bash
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
```
The current PaddlePaddle v3.1.0 does not support NCCL, please check that you have only **1 single GPU** actually enabled or in a pure CPU environment.

### 3.2 Install other dependencies
```bash
pip install -r requirements.txt
```

## 4. Configure API keys

We currently support both OpenAI and Claude APIs. 
Create a `.env` file inside the `base_model/` directory with your credentials:
```bash
# base_model/.env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
UI_TARS_API_KEY=your_ui_tars_api_key_here  # for ui-tars baseline, optional for just running the games
```

## 5. Configure Weights & Biases (wandb)
```bash
wandb login # enter your WandB API key when prompted
```
## 6. Download required models

### Download *OmniParser* models: (preferably)
```bash
python scripts/download_omni_models.py
```

### Download *SAM* models:
  - Download the SAM weights (e.g., `sam_vit_h_4b8939.pth`,`sam_vit_b_01ec64.pth`) from the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) release.
  - Place the file under the root project’s `weights/` folder:
```bash
mkdir -p weights
cd weights
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

### Verify model files
After download completion, your weights directory structure should look like:
```
weights/
├── icon_caption_florence/
│   └── (1 safetensors, 2 json files)
├── icon_detect/
│   └── (1 model.pt, 2 yaml files)
├── sam_vit_b_01ec64.pth
└── sam_vit_h_4b8939.pth # optional
```

# Usage
> **⚠️ Note:** Before running the agent, make sure the target game is already launched and in the main interface. 
> Starting from the main menu may result in excessive exploration of non-essential menus.

## Launch the games from Steam<sup>®</sup>
Download the licensed Slay and Spire, Civilization V from Steam, then launch the game.
For Slay the Spire, select `Play Slay the Spire` option.
For Civilization V, select `Play Sid Meier's Civilization V (DirectX 9)` option.

### Game screen settings
Please adjust the in-game resolution to match the `width` and `height` specified under the `eye` attribute in your `config/*.yaml` configuration files. This ensures the agent observes the window correctly.
| Slay the Spire | Civilization V |
|------------|------------|
| ![Slay the Spire](figs/sts_resolution.jpg) | ![Civilization V](figs/c5_resolution.jpg) |

## Launch the Crafter
```bash
python -m pip install pygame gymnasium crafter
python -m demos.crafter_interactive_launcher --max-steps 1_000_000
```

## Run on Games
### Slay the Spire
```bash
python -m run --config_file "config/sts_explore_claude.yaml"
python -m run --config_file "config/sts_omni_claude.yaml"
python -m run --config_file "config/sts_vector_mcp_claude.yaml"
```

### Civilization V
```bash
python -m run --config_file "config/c5_explore_claude.yaml"
```

## Control
To navigate through the game steps, follow these instructions:
1. When prompted in the terminal, press `[` to initiate a new step.
2. When you see `end action evolution` in the terminal, it signals the end of the current step. At this point, you will need to manually perform some actions in-game to progress to the next step.
3. Press `/` to exit dev mode and save the checkpoint of the current run.
   
## Visualize
To visualize the agent's performance, **open a new terminal**, activate the same conda environment, and run the following command:

```bash
python -m monitor --config_file "config/sts_omni_claude.yaml" --port 8050
```
This will launch a local monitoring server that provides **real-time visualization of the agent's skill trees and library, invocation logs, and action evolution**. By default, the dashboard is available at http://localhost:8050.

You may customize the port by modifying the `--port` argument as needed.

# Results
![Result](figs/result.jpg )
For detailed experimental results and further analysis, please refer to the [full paper](https://arxiv.org/abs/2505.17673).

# Citation
If you find our research useful, please cite it as follows:

```
@article{du2025rethinking,
    title={Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution},
    author={Jiawei Du and Jinlong Wu and Yuzheng Chen and Yucheng Hu and Bing Li and Joey Tianyi Zhou},
    journal={arXiv preprint arXiv:2505.17673},
    year={2025}
}
```

# License
This project is licensed under the MIT License. See LICENSE for details.