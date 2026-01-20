# Osuformer - Beatmap Generator Transformer For Osu

> [!WARNING]
> This is the development branch and it might have bugs or even unfinished code. Please check the releases or main branch for more stable version

# Installation

> [!WARNING]
> This code was only tested on python version `3.13.5`

```
pip install -r requirements.txt
```

# Commands

### Run tests
```
python -m pytest
```

### Update requirements.txt
```
python -m scripts.requirements
```

### Parse osu map
```
python -m scripts.parse_map <path/to/map.osu>
```


# Nice to know


### Dataset cleanup 

I managed to get x3.5 times less by cleaning up with these commands

```
find dataset -type f \( -name "*.jpg" -o -name "*.png" \
  -o -name "*.gif" -o -name "*.mp4" -o -name "*.avi" -o -name "*.flv" \
  -o -name "*.osb" -o -name "*.osk" -o -name "*.osr" \) -delete

find dataset -type d \( -iname "sb" -o -iname "storyboard" \
  -o -iname "skin" -o -iname "skins" -o -iname "effects" \
  -o -iname "particles" -o -iname "bg" -o -iname "video"\
  \) -exec rm -rf {} +
```