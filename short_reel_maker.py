import shutil
import time
from transformers import MusicgenForConditionalGeneration
from transformers import AutoProcessor
import scipy.io.wavfile as wavfile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.tag import pos_tag
import os
from moviepy.editor import *
import numpy as np
import subprocess
import json

# Define the paths to your input and output videos
input_video_path = "vm.mp4"
output_video_path = "output_trailer.mp4"

# Load the input video
clip = VideoFileClip(input_video_path)

# Define the time ranges for the interesting parts you want to trim
#we can add any time clips according to our intrest as document say we can choose to trim any clip in the video
clips_to_extract = [   # 20-30 seconds
    (10, 20),  # 1:20-1:30 minutes
    (200, 220)
]

# Extract the specified clips from the input video
final_clips = [clip.subclip(start, end) for start, end in clips_to_extract]

# Add fade in and fade out transitions to the clips
#i tried to add animation with freezing effect with text overlay result where not that great so 
final_clips_with_transitions = []
for idx, clip in enumerate(final_clips):
    if idx > 0:
        clip = clip.crossfadein(5).crossfadeout(5)
    final_clips_with_transitions.append(clip)

# Concatenate the clips to form the final video
final_video = concatenate_videoclips(final_clips_with_transitions)

# Define resize and padding parameters
#i added black color padding to look nice nothing major problems
RESIZE_WIDTH = 1020
ASPECT_RATIO = 9 / 16  # 9:16 aspect ratio
RESIZE_HEIGHT = int(RESIZE_WIDTH / ASPECT_RATIO)

# Calculate padding size for 2 cm padding at the top and bottom
#i have added 2cm here you can add as your wish
padding_pixels = int(80)  # Fixed padding size

def resize_clip(clip):
    # Resize the clip to the desired dimensions
    resized_clip = clip.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
    
    # Create black padding
    padding_top = np.zeros((padding_pixels, RESIZE_WIDTH, 3), dtype=np.uint8)
    padding_bottom = np.zeros((padding_pixels, RESIZE_WIDTH, 3), dtype=np.uint8)
    
    # Convert numpy arrays to ImageClips
    #i faced many errors for imageclip as i tried to to animation with imageclip every time it gives insufficient memory for np array
    padding_top_clip = ImageClip(padding_top).set_duration(resized_clip.duration)
    padding_bottom_clip = ImageClip(padding_bottom).set_duration(resized_clip.duration)
    
    # Combine clip with padding
    final_clip = CompositeVideoClip([
        resized_clip.set_position(("center", "center")), 
        padding_top_clip.set_position(("center", "top")), 
        padding_bottom_clip.set_position(("center", "bottom"))
    ])
    
    return final_clip

# Resize and add padding to the entire video clip
resized_clip = resize_clip(final_video)

# Write the resized clip to an output file with compatible codecs
resized_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=24, preset="ultrafast", threads=4)

print(f"Resized video created successfully at {output_video_path}")


os.makedirs("output", exist_ok=True)

#i used various linux command to trim code length and perform various functions

command = [
    "whisperx",
    output_video_path,
    "--model",
    "small.en",
    "--output_dir",
    "output",
    "--align_model",
    "WAV2VEC2_ASR_LARGE_LV60K_960H",
    "--compute_type",
    "float32" 
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Command executed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")



def seconds_to_srt_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d},{milliseconds:03d}"



json_file_name = os.path.basename(output_video_path).replace('.mp4', '') + '.json'
json_file_path = os.path.join("output", json_file_name)

with open(json_file_path, "r") as file:
    data = json.load(file)

# Convert JSON data to srt format
srt_data = ""
word_counter = 1
for segment in data["segments"]:
    for word_data in segment["words"]:
        if "start" in word_data and "end" in word_data:
            start_time = seconds_to_srt_time(word_data["start"])
            end_time = seconds_to_srt_time(word_data["end"])
            text = word_data["word"]
            srt_data += f"{word_counter}\n{start_time} --> {end_time}\n{text}\n\n"
            word_counter += 1

srt_data = ""

# Write SRT data to a file

with open("output.srt", "w") as f:
    f.write(srt_data)

print("SRT file generated successfully!")

# Check if the SRT file is not empty
#basically i am doing this because as if video does not have any human speech it will throw so i am handiling like this to skip this command if it doesn't have any human sppech
if os.path.getsize("output.srt") > 0:
    # Define the ffmpeg command
    command = [
        "ffmpeg",
        "-i",
        output_video_path,
        "-vf",
        "subtitles=output.srt",
        "short.avi"
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
else:
    print("SRT file is empty or not generated. Skipping ffmpeg command.")



input_file = "short.avi"
output_file = "output.mp4"

#if above command doen't run this also doesn't run so i am checking if file is present in source directory or not like that and i am using special encodings here which i need specifically when i try to add bacground music thats why i am running command in else part also

if os.path.exists(input_file):
    output_file = "output.mp4"
    command = [
    "ffmpeg",
    "-i", input_file,
    "-vf", "scale=1020:1812",
    "-c:v", "libx264",
    "-b:v", "1000k", 
    "-c:a", "aac",
    "-strict", "experimental",  
    output_file
    ]

    subprocess.run(command)
else:
    input_file="output_trailer.mp4"
    ffmpeg_command = [
    "ffmpeg",
    "-i", input_file,
    "-vf", "scale=1020:1812",
    "-c:v", "libx264",
    "-b:v", "1000k", 
    "-c:a", "aac",
    "-strict", "experimental",  
    output_file
    ]
# Execute the ffmpeg command
    subprocess.run(ffmpeg_command)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

full_transcription = [
    "whisperx",
    input_video_path,
    "--model",
    "small.en",
    "--output_dir",
    "output",
    "--compute_type",
    "float32"
]

try:
    subprocess.run(full_transcription, check=True)
    print("Command executed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")



# Download necessary NLTK datasets
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def analyze_video_description(description):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(description)
    
    mood = 'Positive' if sentiment_scores['compound'] >= 0.05 else 'Negative' if sentiment_scores['compound'] <= -0.05 else 'Neutral'
    
    sentences = nltk.sent_tokenize(description)
    
    scenery = []
    context = []
    
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        
        for word, tag in tagged_words:
            if tag == 'NNP':  # Proper noun, likely to be a scenery or context keyword
                scenery.append(word)
            elif tag == 'NN' or tag == 'NNS':  # Noun or plural noun, can be related to context
                context.append(word)
                
    return {
        'mood': mood,
        'scenery': list(set(scenery)),
        'context': list(set(context))
    }

txt_file_name = os.path.basename(input_video_path).replace('.mp4', '') + '.txt'
txt_file_path = txt_file_name
txt_file_path1 = f"output/{txt_file_path}"

with open(txt_file_path1, "r") as file:
    data = file.read()

analysis = analyze_video_description(data)

mood = analysis['mood'][0] if analysis['mood'] else 'positive'
scenery = analysis['scenery'][0] if analysis['scenery'] else 'pleasant'
context = analysis['context'][0] if analysis['context'] else 'neutral'

# Load model and tokenizer

# Load model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Load processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Tokenize input text
inputs = processor(
    text=["use upbeat/hip-hop based on"+f"Mood: {mood}, Scenery: {scenery}, Context: {context}"],
    padding=True,
    return_tensors="pt",
)

# Generate audio
audio_values = model.generate(**inputs.to("cpu"), do_sample=True, guidance_scale=3, max_new_tokens=512)

# Get sampling rate from model config
sampling_rate = model.config.audio_encoder.sampling_rate

# Save generated audio as a WAV file
wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

subprocess.run(["ffmpeg", "-i", "musicgen_out.wav", "-codec:v", "copy", "output.mp3"])

from moviepy.editor import *
import numpy as np

# Load the 5-second MP3 file
music_clip = AudioFileClip("output.mp3")

# Calculate the number of repetitions needed to match the video duration
video_duration = 120  # Video duration in seconds
music_duration = 5  # Music duration in seconds
num_repetitions = int(np.ceil(video_duration / music_duration))

# Create a list of music clips by repeating the original music clip
music_clips = [music_clip] * num_repetitions

# Concatenate the music clips to create a single music clip
extended_music_clip = concatenate_audioclips(music_clips)

# Fade in and fade out the music to make it more subtle
extended_music_clip = extended_music_clip.audio_fadein(1).audio_fadeout(1)

# Write the extended music clip to the current path
extended_music_clip.write_audiofile("extended_music.mp3")

# Load the video clip
video_clip = VideoFileClip("output.mp4")
audio_clip = AudioFileClip("extended_music.mp3")

# Trim background audio to match video duration
background_audio = audio_clip.set_duration(video_clip.duration)

# Adjust the volume of the background audio
background_audio = background_audio.fx(afx.audio_fadein, 1).fx(afx.volumex, 0.6)

# Overlay the background audio with the main audio
final_audio = CompositeAudioClip([background_audio, video_clip.audio])

# Set the combined audio as the audio for the video clip
video_clip = video_clip.set_audio(final_audio)

# Write the video with the combined audio
video_clip.write_videofile("short_reel.mp4", audio_codec="aac")





try:
    # List of files to delete
    files_to_delete = [
        "short.avi",
        "output.mp4",
        "musicgen_out.wav",
        "output.mp3",
        "output.srt",
        output_video_path
        
    ]
    
    for file in files_to_delete:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Deleted {file}")
            else:
                print(f"{file} does not exist")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
            time.sleep(1)  # Wait for 1 second before the next deletion attempt
        
    # Remove the output directory and its contents
    try:
        shutil.rmtree("output")
        print("Output directory removed successfully!")
    except Exception as e:
        print(f"Error removing output directory: {e}")

except Exception as e:
    print(f"General error: {e}")
