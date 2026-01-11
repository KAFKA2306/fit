import glob
import os
import subprocess
from datetime import datetime

from domain.models import BatchConfig

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


def main():
    config = BatchConfig()
    fbx_files = glob.glob(os.path.join(config.input_dir, "**/*.fbx"), recursive=True)
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for fbx in fbx_files:
        base_name = os.path.splitext(os.path.basename(fbx))[0]
        output_fbx = os.path.join(config.output_dir, f"retargeted_{timestamp}_{base_name}.fbx")
        cmd = [
            config.blender_exe,
            "--background",
            "--python",
            SCRIPT_PATH,
            "--",
            "--base",
            config.base_fbx,
            "--input",
            fbx,
            "--output",
            output_fbx,
            "--config",
            config.config_path,
            "--init-pose",
            config.init_pose,
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
