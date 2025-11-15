import requests
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, vfx
from PIL import Image, ImageDraw, ImageFont
import random

SAMPLE_TRACKS = [
   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3"
]

def get_music_by_mood(mood: str) -> str:
    if "calm" in mood.lower(): return SAMPLE_TRACKS[1]
    if "energetic" in mood.lower(): return SAMPLE_TRACKS[0]
    return random.choice(SAMPLE_TRACKS)

def download_music(url: str, filename="music.mp3") -> str:
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename

def add_text(img_path, text):
    img = Image.open(img_path).convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    W, H = img.size
    font_size = 80
    while font_size > 10:
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        w, _ = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= W - 40: break
        font_size -= 2

    x, y = (W - w) // 2, 50
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0,255))
    draw.text((x, y), text, font=font, fill=(255,255,255,255))

    out = img_path.replace(".jpg", "_caption.png")
    Image.alpha_composite(img, txt_layer).save(out)
    return out

def make_vid(images, mood="energetic"):
    music_file = "music.mp3"

    image_files = [add_text(img, "NEW COLLECTION") for img in images]
    clips = [ImageClip(m).set_duration(5).fadein(1).fadeout(1) for m in image_files]
    video = concatenate_videoclips(clips, method="compose", padding=-1)
    audio = AudioFileClip(music_file).subclip(0, video.duration)
    video = video.set_audio(audio)
    video.write_videofile("fashion_reel_calm.mp4", fps=24)

make_vid(["1.jpg", "2.jpg", "3.jpg", "4.jpg"], mood="calm")
