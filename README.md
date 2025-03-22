# Automated Chorus Detection

![Chorus Prediction](./images/131.webp)

## Project Overview

This project focuses on developing an automated system for detecting choruses in songs using a Convolutional Recurrent Neural Network (CRNN). The model is trained on a custom dataset of 332 annotated songs, predominantly from electronic music genres, and achieved an F1 score of 0.864 (Precision: 0.831, Recall: 0.900) on an unseen test set of 50 songs.

You can try the model in three ways:
1. [Streamlit app on HuggingFace](https://huggingface.co/spaces/dennisvdang/Chorus-Detection)
2. Docker command-line tool (supports YouTube URLs and local audio files)
3. Local installation with virtual environment

## Setup Options

### Option 1: Docker Setup (Recommended)

#### Prerequisites
- Install [Docker](https://www.docker.com/get-started)

#### Building and Running
1. Clone the repository:
   ```bash
   git clone https://github.com/dennisvdang/chorus-detection.git
   cd chorus-detection
   ```

2. Build the Docker image:
   ```bash
   docker build -t chorus-finder .
   ```

3. Run the container:
   
   For interactive mode (choose between YouTube or local file):
   ```bash
   docker run -it chorus-finder
   ```
   
   For local audio files (mount a volume):
   ```bash
   docker run -it -v $(pwd)/input:/app/input chorus-finder python chorus_finder.py --file /app/input/your_song.mp3
   ```

   For YouTube URLs:
   ```bash
   docker run -it chorus-finder python chorus_finder.py --url "https://www.youtube.com/watch?v=your_video_id"
   ```

   Note: YouTube download functionality may be temporarily unavailable due to YouTube's restrictions.

### Option 2: Local Installation with Virtual Environment

#### Prerequisites
- Python 3.11 or later
- FFmpeg
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

#### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/dennisvdang/chorus-detection.git
   cd chorus-detection
   ```

2. Create and activate virtual environment:
   
   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Download the model:
   - Get `best_model_V3.h5` from:
     - [HuggingFace repo](https://huggingface.co/dennisvdang/chorus-detection/tree/main)
     - [Google Drive](https://drive.google.com/file/d/1OoYc5RJwAJo9DmxBHOlP6Qyj1iFvjKoJ/view?usp=sharing)
   - Place in `models/CRNN/` directory

#### Running the CLI Tool

Basic usage (interactive mode):
```bash
python src/chorus_finder.py
```

Analyze YouTube video:
```bash
python src/chorus_finder.py --url "https://www.youtube.com/watch?v=your_video_id"
```

Analyze local audio file:
```bash
python src/chorus_finder.py --file "path/to/your/audio.mp3"
```

Additional options:
```bash
python src/chorus_finder.py --no-plot  # Disable visualization
python src/chorus_finder.py --verbose  # Show detailed progress
```

## Project Resources

Below, you'll find information on where to locate specific files and their purposes:

- [`data/clean_labeled.csv`](data/clean_labeled.csv): The labeled dataset used to train the CRNN.
- [`notebooks/Automated-Chorus-Detection.ipynb`](notebooks/Automated-Chorus-Detection.ipynb): Main development notebook.
- [`notebooks/Preprocessing.ipynb`](notebooks/Preprocessing.ipynb): Audio preprocessing steps.
- [`docs/Data_Annotation_Guide.pdf`](docs/Data_Annotation_Guide.pdf): Manual annotation process guide.
- [`docs/Capstone_Final_Report.pdf`](docs/Capstone_Final_Report.pdf): Detailed project report.
- [`models/CRNN/best_model_V3.h5`](models/CRNN/best_model_V3.h5): Pre-trained CRNN model.

## Project Technical Summary

### Data

The dataset consists of 332 manually labeled songs, predominantly from electronic music genres. Data preparation involved:

1. **Audio preprocessing**: Formatting songs uniformly, processing at a consistent sampling rate, trimming silence, and extracting metadata using Spotify's API. [Link to preprocessing notebook](notebooks/Preprocessing.ipynb)

2. **Manual Chorus Labeling**: Labeling the start and end timestamps of choruses following a set of guidelines. More details on the annotation process can be found in the [Annotation Guide pdf.](docs/Data_Annotation_Guide.pdf)

### Model Preprocessing

- Features such as Root Mean Squared energy, key-invariant chromagrams, Melspectrograms, MFCCs, and tempograms were extracted. These features were decomposed using Non-negative Matrix Factorization using an optimal number of components derived in our exploratory analysis.

- Songs were segmented into timesteps based on musical meters, with positional and grid encoding applied to every audio frame and meter, respectively. Songs and labels were uniformly padded and split into train/validation/test sets, processed into batch sizes of 32 using a custom generator.

- Songs and labels were padded to ensure consistent input lengths for the convolutional layers.

- Data is split into train/validation/test (70/15/15) sets and processed into batch sizes of 32 using a custom generator.

Below are examples of audio feature visualizations of a song with 3 choruses (highlighted in green). The gridlines represent the musical meters, which are used to divide the song into segments; these segments then serve as the timesteps for the CRNN input.

![hspss](./images/hpss.png)
![rms_beat_synced](./images/rms_beat_synced.png)
![chromagram](./images/chromagram_stacked.png)
![tempogram](./images/tempogram.png)

### Modeling

The CRNN model architecture includes:

- Three 1D convolutional layers with ReLU and max-pooling to extract local patterns.
- A Bidirectional LSTM layer to model long-range temporal dependencies.
- A TimeDistributed Dense output layer with sigmoid activation for meter-wise predictions.

``` python
def create_crnn_model(max_frames_per_meter, max_meters, n_features):
    """
    Args:
    max_frames_per_meter (int): Maximum number of frames per meter.
    max_meters (int): Maximum number of meters.
    n_features (int): Number of features per frame.
    """
    frame_input = layers.Input(shape=(max_frames_per_meter, n_features))
    conv1 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(frame_input)
    pool1 = layers.MaxPooling1D(pool_size=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling1D(pool_size=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling1D(pool_size=2, padding='same')(conv3)
    frame_features = layers.Flatten()(pool3)
    frame_feature_model = Model(inputs=frame_input, outputs=frame_features)

    meter_input = layers.Input(shape=(max_meters, max_frames_per_meter, n_features))
    time_distributed = layers.TimeDistributed(frame_feature_model)(meter_input)
    masking_layer = layers.Masking(mask_value=0.0)(time_distributed)
    lstm_out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(masking_layer)
    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(lstm_out)
    model = Model(inputs=meter_input, outputs=output)
    model.compile(optimizer='adam', loss=custom_binary_crossentropy, metrics=[custom_accuracy])
    return model
```

### Training

- Custom loss and accuracy functions handle padded values
- Callbacks to save best model based on minimal validation loss, reduce learning rate on plateau, and early stopping
- Trained for 50 epochs (stopped early after 18 epochs). Training/Validation Loss and Accuracy plotted below:
![Training History](images/training_history.png)

### Results

The model achieved strong results on the held-out test set as shown in the summary table. Visualizations of the predictions on sample test songs are also provided and can be found in the [test_predictions folder](images/test_predictions).

| Metric         | Score  |
|----------------|--------|
| Loss           | 0.278  |
| Accuracy       | 0.891  |
| Precision      | 0.831  |
| Recall         | 0.900  |
| F1 Score       | 0.864  |

![Confusion Matrix](./images/confusion_matrix.png)

## Troubleshooting

### Docker Issues
- If the container fails to build, ensure all required files are present
- For visualization issues, use the `--no-plot` flag
- Mount volumes correctly for local file access

### Virtual Environment Issues
- Ensure FFmpeg is installed and in your system PATH
- Verify the model file is in the correct location
- Check Python version compatibility (3.11+ recommended)

### YouTube Download Issues
- YouTube functionality may be temporarily unavailable due to restrictions
- Try using local audio files as an alternative
- Ensure you have a stable internet connection

If you found this project interesting or informative, feel free to ⭐ star the repository! I welcome any questions, criticisms, or issues you may have.

### Future Plans

I have plans to develop a streamlined pipeline for contributors to preprocess and label their own music data to either train their own custom models or add to the existing dataset to hopefully improve the model's generalizability across various music genres. Stay tuned!
