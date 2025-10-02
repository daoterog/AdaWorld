"""
RTX Dataset Video Extraction Tool

This script processes robotics datasets from the RTX collection, extracting image sequences
from various camera viewpoints and converting them into MP4 video files. It handles multiple
dataset formats and camera configurations commonly found in robotics datasets.
"""

import os
from dataclasses import dataclass
from os import listdir, makedirs, path
from pathlib import Path

# Suppress TensorFlow verbose logging to reduce console output during processing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import imageio  # For video file creation and image processing
import tensorflow_datasets as tfds  # For loading and processing TensorFlow datasets
from tqdm.auto import trange  # For progress bars during batch processing


@dataclass
class Args:
    """Configuration class for dataset processing parameters."""

    save_root: str = Path(__file__).parents[2] / "data" / "open_x"
    orig_root: str = Path(__file__).parents[2] / "data" / "rtx"


def dataset2path(dataset_name) -> str:
    """
    Construct the path to the latest version of a specified dataset.

    RTX datasets are organized with version directories (e.g., '1.0.0', '2.1.0').
    This function automatically selects the latest version based on directory naming.

    Args:
        dataset_name (str): Name of the dataset to locate

    Returns:
        str: Full path to the latest version of the dataset
    """
    print(f"  📁 Looking for dataset: {dataset_name}")

    # List all version directories for the given dataset
    dataset_path = path.join(Args.orig_root, dataset_name)
    versions = listdir(dataset_path)
    versions.sort()
    print(f"     Found versions: {versions}")

    # Filter for version directories (typically have 5-character names like '1.0.0')
    versions = [version for version in versions if len(version) == 5]

    # Select the latest version (last in sorted order)
    version = versions[-1]
    final_path = path.join(Args.orig_root, dataset_name, version)
    print(f"     Using latest version: {version} -> {final_path}")

    return final_path


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    """
    Convert a sequence of images into an MP4 video file.

    Args:
        images (list): List of image arrays (numpy arrays or similar)
        output_file (str): Path where the output video file will be saved
        fps (int, optional): Frames per second for the output video. Defaults to 10.
    """
    print(f"      🎬 Creating video: {output_file} ({len(images)} frames @ {fps}fps)")

    # Create video writer with specified fps
    writer = imageio.get_writer(output_file, fps=fps)

    # Add each image frame to the video
    for image in images:
        writer.append_data(image)

    # Close the writer to finalize the video file
    writer.close()
    print(f"      ✅ Video saved successfully")


def extract_sample(
    tfds_builder,
    obs_key: str,
    dataset_name: str,
    save_dir: str,
    split: str,
    extra: str = None,
) -> None:
    """
    Extract image sequences from a dataset and save them as MP4 videos.

    This function processes episodes from robotics datasets, extracting image observations
    from specified camera viewpoints and converting them to video files.

    Args:
        tfds_builder: TensorFlow dataset builder object
        obs_key (str): Key identifying the observation type (e.g., 'image', 'wrist_image')
        dataset_name (str): Name of the dataset being processed
        save_dir (str): Directory where videos will be saved
        split (str): Dataset split to process ('train', 'test', 'validation')
        extra (str, optional): Additional key for nested observation structures
    """
    print(f"    🔄 Processing {split} split for observation key: {obs_key}")
    if extra:
        print(f"       Using nested key: {extra}")

    # Try to load dataset as TensorFlow dataset, fallback to data source if needed
    try:
        ds = tfds_builder.as_dataset(split=split)
        print(f"       ✅ Loaded as TensorFlow dataset")
    except Exception as e:
        # Some datasets require data source loading instead of dataset loading
        print(
            f"       ⚠️  Failed to load as dataset, trying data source: {str(e)[:100]}..."
        )
        ds = tfds_builder.as_data_source(split=split)
        print(f"       ✅ Loaded as data source")

    ds_iter = iter(ds)
    episodes_processed = 0
    episodes_failed = 0

    # Process each episode in the dataset with progress tracking
    for episode_idx in trange(
        len(ds), desc=f"Extracting {dataset_name.upper()} {split}"
    ):
        try:
            episode = next(ds_iter)

            # Handle different dataset structures for image extraction
            if dataset_name == "robot_vqa":
                # Special handling for robot_vqa dataset which has nested image lists
                images = []
                for step in episode["steps"]:
                    images.extend([img for img in step["observation"][obs_key]])
            elif extra is None:
                # Standard case: extract images directly from observation key
                images = [step["observation"][obs_key] for step in episode["steps"]]
            else:
                # Handle nested observation structures with additional key
                images = [
                    step["observation"][obs_key][extra] for step in episode["steps"]
                ]

            # Convert TensorFlow tensors to numpy arrays if needed
            try:
                images = [image.numpy() for image in images]
            except:
                # Images are already in correct format (numpy arrays)
                images = images

            # Create output path with zero-padded episode index
            save_path = path.join(save_dir, split, f"{episode_idx:08}.mp4")
            makedirs(path.dirname(save_path), exist_ok=True)

            # Convert image sequence to video file
            save_images_to_video(images, save_path)
            episodes_processed += 1

        except Exception as e:
            # Skip episodes that fail to process (corrupted data, missing keys, etc.)
            episodes_failed += 1
            if episodes_failed <= 5:  # Only show first few errors to avoid spam
                print(f"       ⚠️  Episode {episode_idx} failed: {str(e)[:100]}...")
            elif episodes_failed == 6:
                print(f"       ⚠️  Suppressing further error messages...")

    print(
        f"    📊 Split {split} complete: {episodes_processed} processed, {episodes_failed} failed"
    )


dataset_list = [
    "aloha_mobile",
    "asu_table_top_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "berkeley_autolab_ur5",
    "berkeley_cable_routing",
    "berkeley_fanuc_manipulation",
    "berkeley_gnm_cory_hall",
    "berkeley_gnm_recon",
    "berkeley_gnm_sac_son",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "bridge",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "cmu_play_fusion",
    "cmu_playing_with_food",
    "cmu_stretch",
    "columbia_cairlab_pusht_real",
    "conq_hose_manipulation",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dobbe",
    "droid",
    "fmb",
    "fractal20220817_data",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "imperialcollege_sawyer_wrist_cam",
    "io_ai_tech",
    "jaco_play",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "kuka",
    "language_table",
    "maniskill_dataset_converted_externally_to_rlds",
    "mimic_play",
    "nyu_door_opening_surprising_effectiveness",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "plex_robosuite",
    "qut_dexterous_manpulation",
    "robo_net",
    "robo_set",
    "robot_vqa",
    "roboturk",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "taco_play",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "toto",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "uiuc_d3field",
    "utaustin_mutex",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "viola"
]

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
# Track processing statistics
feasible_datasets = 0  # Count of datasets that have compatible image observations
infeasible_datasets = []  # List of datasets that lack compatible image keys
# Comprehensive list of possible image observation keys found across RTX datasets
# These represent different camera viewpoints and image formats used in robotics datasets
display_keys = [
    # Standard image keys
    "image",
    "wrist_image",
    "hand_image",
    "top_image",
    "wrist225_image",
    "wrist45_image",
    "image_manipulation",
    # High-resolution and specialized camera views
    "highres_image",
    "finger_vision_1",
    "finger_vision_2",
    "image_fisheye",
    "wrist_image_left",
    # Multi-view camera setups
    "image_side_1",
    "image_side_2",
    "image_wrist_1",
    "image_wrist_2",
    "image_additional_view",
    "image_left_side",
    "image_right_side",
    "image_left",
    "image_right",
    "image_top",
    "image_wrist",
    # Front and exterior camera views
    "front_image_1",
    "front_image_2",
    "exterior_image_1_left",
    "exterior_image_2_left",
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "hand_color_image",
    # RGB format variations
    "rgb",
    "front_rgb",
    "agentview_rgb",
    "eye_in_hand_rgb",
    "rgb_static",
    "rgb_gripper",
    # Numbered image sequences
    "image_1",
    "image_2",
    "image_3",
    "image_4",
    "image1",
    "image2",
    "images",
    # Camera position identifiers
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]
print("🚀 Starting RTX dataset processing...")
print(f"📁 Source directory: {Args.orig_root}")
print(f"💾 Output directory: {Args.save_root}")
print(f"📋 Datasets to process: {[d for d in dataset_list]}")
print("=" * 80)

# Process each dataset in the list
for dataset_idx, dataset in enumerate(dataset_list, 1):
    print(f"\n🗂️  Processing dataset {dataset_idx}/{len(dataset_list)}: {dataset}")
    is_feasible = False

    try:
        # Load the TensorFlow dataset builder for this dataset
        builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        print(f"  ✅ Dataset builder loaded successfully")
        print(f"  📊 Available splits: {list(builder.info.splits.keys())}")

        # Show available observation keys
        obs_keys = list(builder.info.features["steps"]["observation"].keys())
        print(
            f"  🔍 Available observation keys: {obs_keys[:10]}{'...' if len(obs_keys) > 10 else ''}"
        )

    except Exception as e:
        print(f"  ❌ Failed to load dataset builder: {str(e)}")
        infeasible_datasets.append(dataset)
        continue

    # Check each possible image key to see if it exists in this dataset
    compatible_keys = []
    for display_key in display_keys:
        if display_key in builder.info.features["steps"]["observation"]:
            compatible_keys.append(display_key)

    if compatible_keys:
        print(
            f"  🎯 Found {len(compatible_keys)} compatible image keys: {compatible_keys}"
        )
    else:
        print(f"  ⚠️  No compatible image keys found")
        infeasible_datasets.append(dataset)
        continue

    for display_key in compatible_keys:
        print(f"\n  🖼️  Processing image key: {display_key}")

        # Special handling for mimic_play dataset which has nested image structure
        if dataset == "mimic_play":
            if display_key == "image":
                # Extract front_image_1 from the nested image structure
                folder = path.join(Args.save_root, f"{dataset}-front_image_1")
                if not path.exists(folder):
                    print(f"    📁 Creating output folder: {folder}")
                    for split_name in builder.info.splits.keys():
                        extract_sample(
                            builder,
                            display_key,
                            dataset,
                            folder,
                            split_name,
                            "front_image_1",
                        )
                else:
                    print(f"    ⏭️  Skipping {folder} (already exists)")

                # Extract front_image_2 from the nested image structure
                folder = path.join(Args.save_root, f"{dataset}-front_image_2")
                if not path.exists(folder):
                    print(f"    📁 Creating output folder: {folder}")
                    for split_name in builder.info.splits.keys():
                        extract_sample(
                            builder,
                            display_key,
                            dataset,
                            folder,
                            split_name,
                            "front_image_2",
                        )
                else:
                    print(f"    ⏭️  Skipping {folder} (already exists)")
            else:
                # Handle other image keys in mimic_play with nested structure
                folder = path.join(Args.save_root, f"{dataset}-{display_key}")
                if not path.exists(folder):
                    print(f"    📁 Creating output folder: {folder}")
                    for split_name in builder.info.splits.keys():
                        extract_sample(
                            builder,
                            display_key,
                            dataset,
                            folder,
                            split_name,
                            display_key,
                        )
                else:
                    print(f"    ⏭️  Skipping {folder} (already exists)")
        else:
            # Standard processing for most datasets
            folder = path.join(Args.save_root, f"{dataset}-{display_key}")
            if not path.exists(folder):  # Skip if already processed
                print(f"    📁 Creating output folder: {folder}")
                # Process all available splits (train, test, validation, etc.)
                for split_name in builder.info.splits.keys():
                    extract_sample(builder, display_key, dataset, folder, split_name)
            else:
                print(f"    ⏭️  Skipping {folder} (already exists)")

        # Mark dataset as feasible since it has at least one compatible image key
        is_feasible = True

    # Update processing statistics
    if is_feasible:
        feasible_datasets += 1
        print(f"  ✅ Dataset {dataset} processed successfully")
    else:
        infeasible_datasets.append(dataset)
        print(f"  ❌ Dataset {dataset} marked as infeasible")
# Print final processing summary
print("\n" + "=" * 80)
print("🏁 PROCESSING COMPLETE!")
print("=" * 80)
print(f"✅ Feasible datasets processed: {feasible_datasets}")
print(f"❌ Infeasible datasets: {len(infeasible_datasets)}")

if infeasible_datasets:
    print(f"\n📋 Infeasible datasets list:")
    for i, dataset in enumerate(infeasible_datasets, 1):
        print(f"   {i}. {dataset}")

print(f"\n📁 All output videos saved to: {Args.save_root}")
print("🎉 Done!")
