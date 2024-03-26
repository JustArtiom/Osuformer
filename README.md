# BMG!osu - Beatmap Generator For Osu

> [!WARNING]  
> This is the development branch and it might have bugs or even unfinished code. Please check the releases or main branch for more stable version

BMG!osu is a python project that lets you train and use models different models that help for creating/generating !osu beatmaps.

# The plan...

The plan will be improved with development of the project

### Rythm model

-   Take in parameters such as `audio, bpm, time_signature`
-   Normalise data bringing the audio data to the same **bpm** and normalise **slider speed**
-   Prepare the input data by creating `audio sequences` of the audio file based on the bpm, time signature and offset.
-   Prepare the the output data by creating the `labels` for each audio sequence
-   CNN + RNN model to classify events `(circle, slider_start, slider_hold, slider_end, spin_start, splin_hold, spin_end)`

### HitObjects coordrds generation model

-   For the future

### Slider path generation model

-   For the future
