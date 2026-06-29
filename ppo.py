import argparse

import numpy as np
import riichi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = riichi.VecEnv(args.num_envs, args.seed)

    observation = env.reset()
    done = np.zeros(args.num_envs, dtype=np.bool)

    while not done.all():
        discard = np.argmax(observation, axis=2).astype(np.uint8)
        observation, reward, done = env.step(discard)

    print(np.mean(reward))
