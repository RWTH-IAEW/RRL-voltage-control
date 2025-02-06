import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os

from utils.logger import Logger


class Plotter:

    def __init__(self, logger: Logger, augmented_env):
        self.logger = logger
        self.augmented_env = augmented_env

    def create_video_from_frames(self, frames, episode, fps=30):

        filepath, _ = self.logger.get_folder_relative(os.path.join("videos"))
        filename = os.path.join(filepath, f"{episode}.mp4")

        # First set up the figure
        framewidth, frameheight = frames[0].shape[1], frames[0].shape[0]
        fig, ax = plt.subplots(figsize=(framewidth/100., frameheight/100.), dpi=100)
        ax.axis('off')
        plt.tight_layout(pad=0)

        img = plt.imshow(frames[0])

        def animate(frame):
            img.set_data(frames[frame])
            return [img]

        anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
        plt.close()
        anim.save(filename, writer="ffmpeg", fps=fps)

        return filename
