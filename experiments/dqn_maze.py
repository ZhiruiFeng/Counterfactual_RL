import sys
sys.path.append("../")
import __init__ as deepq
import maze

def main():
    env = maze.MazeSMDP()
    model = deepq.models.cnn_to_mlp(
        convs=[(32,3,1),(32,3,1),(64,4,2)],
        hiddens=[64],
        dueling=False
    )
    deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False
        #prioritized_replay=bool(args.prioritized),
        #prioritized_replay_alpha=args.prioritized_replay_alpha
    )

if __name__ == '__main__':
    main()
