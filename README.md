# osu!BMG - Beatmap Generator For Osu

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