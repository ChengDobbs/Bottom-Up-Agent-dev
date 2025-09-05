import yaml
import argparse

from BottomUpAgent.Detector import Detector
from BottomUpAgent.Brain import Brain
from BottomUpAgent.Eye import Eye
from BottomUpAgent.Mcts import MCST
from BottomUpAgent.BottomUpAgent import BottomUpAgent



def explore_action(config, action_id):
    eye = Eye(config)
    brain = Brain(config)
    detector = Detector(config)

    sup = BottomUpAgent(config)

    screen = eye.get_screenshot_cv()
    state_feature = detector.encode_image(screen)
    state = {"state_feature": state_feature, "screen": screen}

    action = brain.long_memory.get_action(action_id)

    potential_action_clusters = brain.long_memory.get_action_clusters(state['state_feature'])

    sup.explore_mutation("play the game", action, potential_action_clusters)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, help='path to the config file')
    parser.add_argument('--action_id', required=True, help='action id to explore')

    opt = parser.parse_args()

    with open(opt.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    explore_action(config, opt.action_id)

   

