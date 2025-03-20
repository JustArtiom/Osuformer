import os, csv, requests, shutil, re
from collections import Counter
from time import sleep
from utils import args

config = args.generate_dataset()

OSU_SONGS_PATH = config.songs_path
OSU_API_KEY = config.api_key
OUTPUT = config.out_path
MIN = config.min
MAX = config.max
MINUR = config.min_ur

if not OSU_API_KEY: raise Exception("You do not have a osu api key set.")

folders = [f for f in os.listdir(OSU_SONGS_PATH) if os.path.isdir(os.path.join(OSU_SONGS_PATH, f))][::-1]
map_ids = [int(f.split(' ')[0]) for f in folders if f.split(' ')[0].isdigit()] # Filter only the on

csv_data = [["MAP ID", "DIFFICULTY ID", "MAP PATH", "AUDIO PATH", "LENGTH", "DIFFICULTY NAME", "DIFFICULTY RATING", "BPM", "USER RATING"]]

def findAudioFileName(folder_path):
  audio_filenames = []

  pattern = re.compile(r"AudioFilename:\s*(.*)")

  for filename in os.listdir(folder_path):
    if filename.endswith(".osu"):
      file_path = os.path.join(folder_path, filename)

      with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

        match = pattern.search(content)
        if match:
          audio_filenames.append(match.group(1)) 

  if not audio_filenames:
    return None

  counter = Counter(audio_filenames)

  most_common_audio = counter.most_common(1)[0][0]
  return most_common_audio, most_common_audio.split(".")[-1], len(set(audio_filenames))

def osufiledata(file_path: str):
  bmid_pattern = re.compile(r"BeatmapID:\s*(\d+)")

  with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
      match_bmid = bmid_pattern.search(line)
      if match_bmid:
        return int(match_bmid.group(1))

  return None

def read(file_path: str) -> str:
  try:
    with open(file_path, "r", encoding="utf-8") as file:
      return file.read()
  except Exception:
    return ""

def doMap(mapid: int):
  folder = [f for f in folders if f.startswith(str(mapid))][0]
  folder = os.path.join(OSU_SONGS_PATH, folder)
  response = requests.get(f"https://osu.ppy.sh/api/get_beatmaps?k={OSU_API_KEY}&s={mapid}&m=0")
  if response.status_code == 200:
    diffs = response.json() or []
    if(not len(diffs)): return
    audiofname, format, repetitions = findAudioFileName(folder)
    if(repetitions > 1): 
      print(f"The map {mapid} has more than one audio files")
      return
    aud_dest = os.path.join(OUTPUT, "audios", f"{mapid}.{format}")
    savedaud = False

    for diff in diffs:
      if diff["mode"] != "0": continue
      dr = float(diff["difficultyrating"])
      ur = float(diff["rating"])
      if dr < MIN or dr > MAX or ur < MINUR: continue
      version = diff["version"]
      bmid = diff["beatmap_id"]
      if(not savedaud):
        try:
          shutil.copy(os.path.join(folder, audiofname), aud_dest)
        except Exception as e:
          print(e)
          return
        savedaud = True
    
      vfname = [j for j in os.listdir(folder) if version in j and j.endswith(".osu")]
      if(not len (vfname)): continue
      else: 
        if(len(vfname) == 1):
          vfname = vfname[0]
        else: 
          sadasd = None
          for vfn in vfname:
            bmidtmp = osufiledata(os.path.join(folder, vfn))
            if int(bmid) == bmidtmp:
              sadasd = vfn

          if sadasd:
            vfname = sadasd
          else: return
            

      osu_dest = os.path.join(OUTPUT, "maps", f"{bmid}.osu")
      try:
        shutil.copy(os.path.join(folder, vfname), osu_dest)
      except Exception as e:
        print(e)
        return
      csv_data.append([mapid, bmid, osu_dest, aud_dest, diff["total_length"], version, diff["difficultyrating"], diff["bpm"], diff["rating"]])
  elif response.status_code == 401:
    raise Exception("Invalid OSU! API KEY")
  else:
    print("Error Fetching Map ?")

if os.path.exists(OUTPUT):
  a = input("OVEWRITING THE CURRENT FOLDER? (y)/n: ")
  if(a != 'n'):
    shutil.rmtree(OUTPUT)
    
os.makedirs(OUTPUT)
os.makedirs(os.path.join(OUTPUT, "audios"))
os.makedirs(os.path.join(OUTPUT, "maps"))

for index, map_id in enumerate(map_ids):
  print(f"{index} Fetching {map_id}")
  doMap(map_id)
  sleep(.1)

  if(index % 10 == 0):
    print("Saving...")
    with open(os.path.join(OUTPUT, "info.csv"), mode='w', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      writer.writerows(csv_data)