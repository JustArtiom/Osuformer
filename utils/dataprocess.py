class OsuParsedData:
    def __init__(self):
        self.general = {}
        self.metadata = {}
        self.difficulty = {}
        self.timingpoints = []
        self.hitobjects = []


def try_to_nr(nr):
    if nr.isdigit():
        return int(nr)
    else:
        try:
            return float(nr)
        except Exception:
            return nr


def closest_lower_value_tp(value: int, numbers: list[int]) -> int:
    closest = None
    needed = None

    for num in numbers:
        if num[0] < value:
            if closest is None or num[0] > closest:
                closest = num[0]
                needed = num

    return needed


def keyword_to_obj(str: str):
    key, value = str.split(":", 1)
    return key.strip(), try_to_nr(value.strip())


def parse_osu_file(osu_file_path: str) -> OsuParsedData:
    """
    Parse an osu! file into an object.

    This function reads an osu! file and parses its contents into an appropriate object
    representing the data in the file.

    Args:
        osu_file_path (str): The path to the osu! file to be parsed.

    Returns:
        OsuPrsedData: An object containing the parsed data from the osu! file.
    """

    parsed_data = OsuParsedData()
    current_header = ""

    with open(osu_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_header = line.lower().strip('[]')
                continue

            if current_header in ["general", "metadata", "difficulty"]:
                key, value = keyword_to_obj(line)
                getattr(parsed_data, current_header)[key] = value
            elif current_header in ["timingpoints", "hitobjects"]:
                line_vals = []

                for val in line.strip().split(","):
                    val = try_to_nr(val)
                    line_vals.append(val)

                getattr(parsed_data, current_header).append(line_vals)

    return parsed_data


def parse_osu_files(osu_files: list[str]):
    maps_data: list[OsuParsedData] = []

    for i in range(0, len(osu_files)):
        maps_data.append(parse_osu_file(osu_files[i]))

    return maps_data


def timing_to_bpm(timing_value: float) -> float:
    """
    Convert timing value to BPM.

    Args:
        timing_value (float): The timing value in milliseconds.

    Returns:
        float: The corresponding BPM.
    """
    return round(60000 / timing_value, 2)


def get_rhythm_beats(length: int, beat_length: float, signature: int, offset: float, rounded: bool = True):
    sub_beat_length = beat_length / signature
    beats = []
    i = offset
    while i <= length:
        beats.append(round(i) if rounded else i)
        i += sub_beat_length
    return beats, sub_beat_length, offset


def filter_timing_points(timingpoints: list[int | str]):
    inhired: list[int | str] = []
    uninhired: list[int | str] = []

    for tp in timingpoints:
        if (tp[6] == 1):
            inhired.append(tp)
        else:
            uninhired.append(tp)

    return inhired, uninhired


def get_maps_rhythm_beats(maps_data: list[OsuParsedData], signature: int, length: list[int]):
    arr: list[int] = []
    timing: list[int] = []

    for i, map in enumerate(maps_data):
        inhired_points, _ = filter_timing_points(map.timingpoints)
        rhythm, sub_beat_length, offset = get_rhythm_beats(
            length=length[i],
            beat_length=inhired_points[0][1],
            signature=signature,
            offset=inhired_points[0][0]
        )
        arr.append(rhythm)
        timing.append([sub_beat_length, offset])

    return arr, timing


def snap_to_closest(time: int, rhythm_beats: list[int]) -> int:
    closest_beat = rhythm_beats[0]
    min_diff = abs(time - closest_beat)

    for beat in rhythm_beats:
        diff = abs(time - beat)
        if diff < min_diff:
            min_diff = diff
            closest_beat = beat

    return closest_beat, min_diff


def _maps_filter_spinners(maps_data: list[OsuParsedData]):
    for map_i in range(len(maps_data)):
        filtered_hitobjects = []

        for ho_i in range(len(maps_data[map_i].hitobjects)):
            ho = maps_data[map_i].hitobjects[ho_i]
            if len(ho) != 7:
                filtered_hitobjects.append(ho)

        maps_data[map_i].hitobjects = filtered_hitobjects


def _snap_hit_objects_timing(maps_data: list[OsuParsedData], rhythm_beats: list[int]):
    for map_i in range(len(maps_data)):
        map_title = maps_data[map_i].metadata["Title"]
        map_by = maps_data[map_i].metadata["Artist"]

        for ho_i in range(len(maps_data[map_i].hitobjects)):
            time, difference = snap_to_closest(
                maps_data[map_i].hitobjects[ho_i][2],
                rhythm_beats[map_i]
            )

            if (difference > 10):
                raise ValueError(
                    "The signature beat difference is too big. Map: " +
                    map_title + " by " + map_by +
                    ", Expected <= 10, got: " + str(difference) + " ms " +
                    "at time " + str(maps_data[map_i].hitobjects[ho_i][2]) + "ms. " +
                    "Try using a bigger time signature")

            maps_data[map_i].hitobjects[ho_i][2] = time


def _filter_variable_bpm(maps_data: list[OsuParsedData], audio_data: list):
    filtered_maps = []
    filtered_audios = []

    for map_i in range(len(maps_data)):
        inhired_points, _ = filter_timing_points(
            maps_data[map_i].timingpoints)

        if len(set(i[1] for i in inhired_points)) == 1:
            print(
                "Adding \""+maps_data[map_i].metadata["Title"] + "\" for processing")
            filtered_maps.append(maps_data[map_i])
            filtered_audios.append(audio_data[map_i])
        else:
            print("Warning: Map " +
                  maps_data[map_i].metadata['Title'] + " has been filtered because of variable bpm")

    maps_data[:] = filtered_maps
    audio_data[:] = filtered_audios


def _default_slider_multiplier(maps_data: list[OsuParsedData]):
    for map_i in range(len(maps_data)):
        slider_multi = maps_data[map_i].difficulty["SliderMultiplier"]

        for tp_i in range(len(maps_data[map_i].timingpoints)):
            tp = maps_data[map_i].timingpoints[tp_i]
            if (tp[6] == 1):
                continue

            maps_data[map_i].timingpoints[tp_i][1] = tp[1] * slider_multi

        maps_data[map_i].difficulty["SliderMultiplier"] = 1


def get_maps_song_intensity_and_volume(maps_data: list[OsuParsedData], rythm_maps: list[int]):
    intensity_data: list[int] = []
    volume_data: list[int] = []

    for map_i in range(len(maps_data)):
        vdata = []
        idata = []
        _, tp = filter_timing_points(maps_data[map_i].timingpoints)

        for beat in rythm_maps[map_i]:
            tp_needed = closest_lower_value_tp(beat, tp) or tp[0]
            idata.append(abs(tp_needed[1]))
            vdata.append(tp_needed[5])

        intensity_data.append(idata)
        volume_data.append(vdata)

    return intensity_data, volume_data


def get_preprocessed_hit_objects(maps_data: list[OsuParsedData], rhythm_maps: list[int]):
    maps = []

    # [time, nothing, circle, slider]
    for map_i in range(len(maps_data)):
        new_ho: list = []
        hitobjects = maps_data[map_i].hitobjects
        rhythm = rhythm_maps[map_i]

        for beat in rhythm:
            ho = [i for i in hitobjects if i[2] == beat]

            if (len(ho) == 0):
                ho = []
            else:
                ho = ho[0]

            if (len(ho) == 8):
                new_ho.append([ho[2], 0, 0, 1])
            elif (len(ho) == 6):
                new_ho.append([ho[2], 0, 1, 0])
            else:
                new_ho.append([beat, 1, 0, 0])
        maps.append(new_ho)

    return maps
