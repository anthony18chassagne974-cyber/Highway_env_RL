from pathlib import Path
import base64
import imageio.v2 as imageio
import tempfile
from IPython.display import HTML

def save_video(frames, output_path, fps=20):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=None)
    return output_path

def show_video_file(video_path, width=600):
    with open(video_path, "rb") as f:
        video = f.read()
    data_url = "data:video/mp4;base64," + base64.b64encode(video).decode()
    return HTML(f"""
    <video width="{width}" controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """)

def save_temp_video_and_show(frames, fps=20, width=600):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
    imageio.mimsave(temp_path, frames, fps=fps, macro_block_size=None)
    return show_video_file(temp_path, width=width)
