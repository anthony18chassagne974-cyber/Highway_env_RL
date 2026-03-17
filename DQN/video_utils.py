import numpy as np


def record_episode(env, policy=None, max_steps=200):

    frames = []
    obs, info = env.reset()

    for _ in range(max_steps):

        frame = env.render()

        # IMPORTANT : vérifier que la frame existe
        if frame is not None:
            frames.append(frame)

        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(obs)

        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break

    return frames

import base64
import imageio.v2 as imageio
import tempfile
from IPython.display import HTML


def show_video(frames, fps=20, width=600):
    """
    Affiche une vidéo directement dans le notebook à partir d'une liste de frames
    """

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        video_path = tmp.name

    imageio.mimsave(
        video_path,
        frames,
        fps=fps,
        macro_block_size=None
    )

    with open(video_path, "rb") as f:
        video = f.read()

    data_url = "data:video/mp4;base64," + base64.b64encode(video).decode()

    return HTML(f"""
    <video width="{width}" controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)