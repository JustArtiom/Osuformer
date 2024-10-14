from keras import layers, models
from utils import args
from utils.audioprocess import preprocess_audio_files
from utils.working_space import WorkingSpace
from utils.dataset import Dataset
from utils.dataprocess import get_maps_song_intensity_and_volume, parse_osu_files, get_maps_rhythm_beats, _filter_variable_bpm, _snap_hit_objects_timing, _maps_filter_spinners, _default_slider_multiplier, get_preprocessed_hit_objects

config = args.parse()

SIGNATURE = 4  # 1/4 Time signature
SAMPLE_RATE = 16000
N_MELS = 64
RANGE = 100

workingSpace = WorkingSpace(config.out_path)
dataset = Dataset(config.train_path)

audios_files, maps_files = dataset.getFiles()

print(f"Loaded {len(audios_files)} audios and {len(maps_files)} maps")
if (len(audios_files) != len(maps_files)):
    raise RuntimeError(
        "The length of audio files is not equal to the collected osu maps")

maps_data = parse_osu_files(maps_files)

_filter_variable_bpm(maps_data, audios_files)
_maps_filter_spinners(maps_data)  # everyone hates those..

audios_data, audios_duration = preprocess_audio_files(
    audios_files, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
maps_rhythm_beats, maps_timing = get_maps_rhythm_beats(
    maps_data, SIGNATURE, audios_duration)

_snap_hit_objects_timing(maps_data, maps_rhythm_beats)
_default_slider_multiplier(maps_data)

hit_objects = get_preprocessed_hit_objects(maps_data, maps_rhythm_beats)
maps_intensity_data, maps_volume_data = get_maps_song_intensity_and_volume(
    maps_data, maps_rhythm_beats)

print("")
print("Training data loaded successfully!")
print(f"Loaded {len(maps_data)} map and {len(audios_data)} audio files")
print(
    "These all should be equal:",
    len(maps_data),
    len(audios_data),
    len(hit_objects),
    len(maps_intensity_data),
    len(maps_volume_data),
    len(maps_rhythm_beats),
    len(maps_timing)
)
print(f"Maps were loaded with 1/{SIGNATURE} signature")
print(f"Audios were loaded with {SAMPLE_RATE}Hz and {N_MELS} n_mels")
print("")

X_audio, X_features, Y_labels, Y_features = dataset.get_data_prepared(
    audios_data=audios_data,
    maps_beats=maps_rhythm_beats,
    maps_hit_objects=hit_objects,
    maps_intensity_data=maps_intensity_data,
    maps_volume_data=maps_volume_data,
    collect_range=RANGE
)

print("Shapes:")
print(X_audio.shape)
print(X_features.shape)
print(Y_labels.shape)
print(Y_features.shape)
print("")


# Reshape your audio data to have 1 channel
X_audio_reshaped = X_audio.reshape(
    # Shape: (9403, 64, 200, 1)
    X_audio.shape[0], X_audio.shape[1], X_audio.shape[2], 1)

# Input layers
input_audio = layers.Input(
    shape=(64, 200, 1))  # Shape: (64, 200)
input_metadata = layers.Input(shape=(X_features.shape[1],))  # Shape: (2,)

# Use Conv1D instead of Conv2D for temporal data
cnn = layers.TimeDistributed(layers.Conv1D(
    32, kernel_size=3, activation='relu'))(input_audio)
cnn = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(cnn)
cnn = layers.TimeDistributed(layers.Conv1D(
    64, kernel_size=3, activation='relu'))(cnn)
cnn = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(cnn)
cnn = layers.TimeDistributed(layers.Flatten())(cnn)

# LSTM part
lstm = layers.LSTM(128, return_sequences=False)(cnn)

# Combine CNN and metadata
combined = layers.Concatenate()([lstm, input_metadata])

# Dense layers
dense1 = layers.Dense(128, activation='relu')(combined)
dense2 = layers.Dense(64, activation='relu')(dense1)

# Outputs
output_action = layers.Dense(3, activation='sigmoid')(dense2)  # Action output
output_intensity = layers.Dense(
    2, activation='linear')(dense2)  # Intensity output

# Create model
model = models.Model(inputs=[input_audio, input_metadata], outputs=[
                     output_action, output_intensity])

# Compile the model with metrics for each output
model.compile(optimizer='adam',
              loss=['binary_crossentropy', 'mse'],
              metrics=[['accuracy'], ['accuracy']])  # Metrics for each output


# Fit the model
model.fit(x=[X_audio, X_features], y=[
          Y_labels, Y_features], batch_size=16, epochs=50)
