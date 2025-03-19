from . import tools
from typing import Union

class General:
    AudioFilename: str
    AudioLeadIn: int
    PreviewTime: int
    Countdown: int
    SampleSet: str
    StackLeniency: int
    Mode: int
    LetterboxInBreaks: int

    def __repr__(self):
        return (f"General(AudioFilename={self.AudioFilename}, AudioLeadIn={self.AudioLeadIn}, "
                f"PreviewTime={self.PreviewTime}, Countdown={self.Countdown}, "
                f"SampleSet={self.SampleSet}, StackLeniency={self.StackLeniency}, "
                f"Mode={self.Mode}, LetterboxInBreaks={self.LetterboxInBreaks})")


class Metadata:
    Title: str
    TitleUnicode: str
    Artist: str
    ArtistUnicode: str
    Creator: str
    Version: str
    Source: str
    Tags: str
    BeatmapID: int
    BeatmapSetID: int

    def __repr__(self):
        return (f"Metadata(Title={self.Title}, TitleUnicode={self.TitleUnicode}, "
                f"Artist={self.Artist}, ArtistUnicode={self.ArtistUnicode}, "
                f"Creator={self.Creator}, Version={self.Version}, "
                f"Source={self.Source}, Tags={self.Tags}, "
                f"BeatmapID={self.BeatmapID}, BeatmapSetID={self.BeatmapSetID})")


class Difficulty:
    HPDrainRate: int
    CircleSize: int
    OverallDifficulty: int
    ApproachRate: int
    SliderMultiplier: int
    SliderTickRate: int
        
    def __repr__(self):
        return (f"Difficulty(HPDrainRate={self.HPDrainRate}, CircleSize={self.CircleSize}, "
                f"OverallDifficulty={self.OverallDifficulty}, ApproachRate={self.ApproachRate}, "
                f"SliderMultiplier={self.SliderMultiplier}, SliderTickRate={self.SliderTickRate})")

class TimingPoint:
    def __init__(self, v: str):
        v = v.split(",")
        self.time = int(v[0])
        self.beatLength = float(v[1])
        self.meter = int(v[2])
        self.sampleSet = int(v[3])
        self.sampleIndex = int(v[4])
        self.volume = int(v[5])
        self.uninherited = int(v[6])
        self.effects = int(v[7])
    
    def __repr__(self):
        return (f"TimingPoint(time={self.time}, beatLength={self.beatLength}, meter={self.meter}, "
                f"sampleSet={self.sampleSet}, sampleIndex={self.sampleIndex}, volume={self.volume}, "
                f"uninherited={self.uninherited}, effects={self.effects})")


class HitSample:
    def __init__(self, v: str):
        v = v.split(":")
        self.normalSet = int(v[0])
        self.additionSet = int(v[1])
        self.index = int(v[2])
        self.volume = int(v[3])
        self.filename = v[4] or None

    def __repr__(self):
        return f"HitSample(normalSet={self.normalSet}, additionSet={self.additionSet}, index={self.index}, volume={self.volume}, filename={self.filename})"


class Point():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

class Spinner:
    def __init__(self, v: str):
        v = v.split(",")
        self.time = int(v[2])
        self.type = int(v[3])
        self.hitSound = int(v[4])
        self.endTime = int(v[5])
        self.hitSample = HitSample(v[6])
    
    def __repr__(self):
        return (f"Spinner(time={self.time}, type={self.type}), "
                f"hitSound={self.hitSound}, endTime={self.endTime}, "
                f"hitSample={self.hitSample})")


class Circle(Point):
    def __init__(self, v: str):
        v = v.split(",")
        super().__init__(v[0], v[1])
        self.time = int(v[2])
        self.type = int(v[3])
        self.hitSound = int(v[4])
        self.hitSample = HitSample(v[5])

    def __repr__(self):
        return f"Circle(x={self.x}, y={self.y}, time={self.time}, type={self.type}, hitSound={self.hitSound}, hitSample={self.hitSample})"

class Curve:
    def __init__(self, v: str):
        v = v.split("|")
        self.path = [(Point(*i.split(":")) if ":" in i else i) for i in v]
    
    def __repr__(self):
        return f"Curve(path={self.path})"

class Slider(Circle):
    def __init__(self, v: str):
        v = v.split(",")
        super().__init__(",".join([*v[:5], v[10] if len(v) > 10 else "0:0:0:0:"]))
        self.curve = Curve(v[5])
        self.slides = int(v[6])
        self.length = float(v[7])
        self.edgeSounds = v[9] if len(v) > 9 else None
        self.edgeSets = v[10] if len(v) > 10 else None

    def __repr__(self):
        return f"Slider(x={self.x}, y={self.y}, time={self.time}, type={self.type}, hitSound={self.hitSound}, curve={self.curve}, slides={self.slides}, length={self.length}, edgeSounds={self.edgeSounds}, edgeSets={self.edgeSets}, hitSample={self.hitSample}, hitSample={self.hitSample})"


class Osu:
    General = General()
    Metadata = Metadata()
    Difficulty = Difficulty()
    TimingPoints = []
    HitObjects: list[Union[Slider, Circle, Spinner]] = []

    def __init__(self, file_data: str):
        current_header = ""
        for line in file_data:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_header = line.strip('[]')
                continue

            if current_header in ["General", "Metadata", "Difficulty"]:
                key, value = self.keyword_to_obj(line)
                setattr(getattr(self, current_header), key, value)
            elif current_header == "TimingPoints":
                self.TimingPoints.append(TimingPoint(line))
            elif current_header == "HitObjects":
                line_len = len(line.split(","))

                if(line_len == 7): # Spinner
                    self.HitObjects.append(Spinner(line))
                elif(line_len == 6): # Circle
                    self.HitObjects.append(Circle(line))
                elif(line_len == 11 or line_len == 8): # Slider
                    self.HitObjects.append(Slider(line))
                else:
                    print("Class Osu: Unknown HitObject Detected: "+line)

    def keyword_to_obj(self, str: str):
        key, value = str.split(":", 1)
        return key.strip(), tools.try_to_nr(value.strip())
        
    def __repr__(self):
        return (f"Osu(General={self.General}, Metadata={self.Metadata}, Difficulty={self.Difficulty}, "
                f"TimingPoints={self.TimingPoints}, HitObjects={self.HitObjects})")

    @staticmethod
    def from_file(path: str):
        with open(path, 'r', encoding='utf-8') as file:
            return Osu(file)