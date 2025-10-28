import argparse
import os
import random
from pathlib import Path

import imageio
import retro
from tqdm.auto import trange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Retro Environments")
    parser.add_argument(
        "--num_logs", type=int, default=100, help="Number of episodes to generate"
    )
    parser.add_argument(
        "--timeout", type=int, default=1000, help="Timeout for generating samples"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=Path(__file__).parents[1],
        help="Root folder to save the videos",
    )
    args = parser.parse_args()
    return args


def save_images_to_video(images: list, output_file: Path, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(
    env_name: str, timeout: int, root: str, split: Path, bias: int
) -> None:
    env = retro.make(game=env_name, render_mode="rgb_array")

    frames = [env.reset()[0]]
    for t in range(timeout - 1):
        bias += t // 500
        action_todo = env.action_space.sample()
        if random.random() > 0.1 and bias < 4:
            action_todo[4 + bias] = 1

        obs, reward, terminated, truncated, info = env.step(action_todo)  # 60 FPS
        frames.append(obs)
        if terminated:
            frames.append(env.reset()[0])

    env.close()

    save_dir = Path(root) / "retro" / env_name / split
    save_dir.mkdir(parents=True, exist_ok=True)
    current_idx = len(os.listdir(save_dir))
    save_path = save_dir / f"{current_idx:05}.mp4"

    save_images_to_video(frames, save_path)


def main():

    args = parse_args()

    env_list = retro.data.list_games()
    for env_name in env_list:
        try:
            for n_log in trange(
                args.num_logs,
                desc=f"Generating {args.num_logs} {env_name.upper()} videos for training",
            ):
                generate_sample(env_name, args.timeout, args.root, "train", n_log % 5)

            for n_log in trange(
                args.num_logs // 10,
                desc=f"Generating {args.num_logs // 10} {env_name.upper()} videos for test",
            ):
                generate_sample(env_name, args.timeout, args.root, "test", n_log % 5)
        except:
            pass


if __name__ == "__main__":
    main()
