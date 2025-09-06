import yaml
import argparse
from BottomUpAgent.BottomUpAgent import BottomUpAgent
from BottomUpAgent.GymAgent import GymBottomUpAgent


def main(config, interactive=False):
    # Check if this is a Gym environment configuration
    if 'gym' in config or config.get('game_name') in ['Crafter', 'CrafterReward-v1', 'CartPole-v1', 'LunarLander-v2']:
        print(f"Creating GymBottomUpAgent for {config.get('game_name', 'Unknown')}")
        gamer = GymBottomUpAgent(config)
    else:
        print(f"Creating standard BottomUpAgent for {config.get('game_name', 'Unknown')}")
        gamer = BottomUpAgent(config)
    
    task = 'Play the game'
    
    # For Gym environments, choose between interactive and standard mode
    if isinstance(gamer, GymBottomUpAgent):
        if interactive:
            print("Running Interactive BottomUp Mode with GUI Control...")
            try:
                # Run interactive mode with GUI control and BottomUp analysis
                episode_stats = gamer.run_interactive()
                print(f"Interactive session completed: {episode_stats['steps']} steps, {episode_stats['total_reward']:.2f} reward")
            except Exception as e:
                print(f"Error during interactive session: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gamer.close()
        else:
            print("Running Gym environment with BottomUpAgent framework...")
            try:
                # Run a single episode to demonstrate the integration
                episode_stats = gamer.run_episode()
                print(f"Episode completed: {episode_stats['steps']} steps, {episode_stats['total_reward']:.2f} reward")
            except Exception as e:
                print(f"Error during Gym episode: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gamer.close()
    else:
        # Standard BottomUpAgent run method
        gamer.run(task=task, max_step=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, help='path to the config file')
    
    opt = parser.parse_args()

    with open(opt.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config=config)