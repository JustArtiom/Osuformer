BEAT_DICT = {
    "None": 0,
    "Circle": 1,
    "Slider": 2,
    "Sliding": 3,
    "SliderEnd": 4,
    "Spinner": 5
}

def try_to_nr(nr: str):
    if nr.isdigit():
        return int(nr)
    else:
        try:
            return float(nr)
        except Exception:
            return nr
