import argparse
from pathlib import Path

import imageio
from gym3 import types_np
from procgen import ProcgenGym3Env
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Procgen Environments")
    parser.add_argument(
        "--num_logs", type=int, default=10000, help="Number of episodes to generate"
    )
    parser.add_argument(
        "--timeout", type=int, default=1000, help="Timeout for generating samples"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parents[1],
        help="Root folder to save the videos",
    )
    args = parser.parse_args()
    return args


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    writer = imageio.get_writer(output_file, fps=fps)
    for image in images:
        writer.append_data(image)
    writer.close()


def generate_sample(
    env_name: str, start_level: int, timeout: int, root: Path, split: str
) -> None:
    env = ProcgenGym3Env(
        env_name=env_name,
        num=1,
        num_levels=1,
        start_level=start_level,
        use_sequential_levels=False,
        distribution_mode="hard",
        render_mode="rgb_array",
    )

    frames = [env.get_info()[0]["rgb"]]
    for _ in range(timeout - 1):
        action_todo = types_np.sample(env.ac_space, bshape=(env.num,))
        env.act(action_todo)  # 15 FPS
        frames.append(env.get_info()[0]["rgb"])

    env.close()

    save_path = root / "procgen" / env_name / split / f"{start_level:05}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_images_to_video(frames, save_path)


def main() -> None:

    args = parse_args()

    env_list = [
        "bigfish",
        "bossfight",
        "caveflyer",
        "chaser",
        "climber",
        "coinrun",
        "dodgeball",
        "fruitbot",
        "heist",
        "jumper",
        "leaper",
        "maze",
        "miner",
        "ninja",
        "plunder",
        "starpilot",
    ]
    for env_name in env_list:
        for i in tqdm(
            range(args.num_logs // 10, args.num_logs),
            desc=f"Generating {args.num_logs // 10 * 9} {env_name.upper()} videos for training",
        ):
            generate_sample(env_name, i, args.timeout, args.root, "train")

        for i in tqdm(
            range(args.num_logs // 10),
            desc=f"Generating {args.num_logs // 10} {env_name.upper()} videos for test",
        ):
            generate_sample(env_name, i, args.timeout, args.root, "test")


if __name__ == "__main__":
    main()
