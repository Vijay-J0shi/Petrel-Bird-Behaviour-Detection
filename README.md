# Snow Petrel Behavior Pattern Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Technologies](#key-technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Project Overview
Snow Petrels are important sentinel species in Antarctica, making their behavior patterns critical for ecological monitoring and research. This project uses video data containing noisy images to predict behaviors of Snow Petrels by first applying pose estimation and then using ST-GCN (Spatio-Temporal Graph Convolutional Networks) to analyze the spatio-temporal relationships between keypoints over time.

## Objectives
The primary goal of this project is to:

- Detect, classify, and predict the behavioral patterns of Snow Petrels using deep learning methods.
- Model these behaviors based on pose estimation data by applying Spatio-Temporal Graph Convolutional Networks (ST-GCN).

## Dataset
- **Data Source**: The dataset consists of approximately 400,000 frames, each 144x40 pixels in size, captured from a Snow Petrel's nest inside a cave in Antarctica.
- **Data Characteristics**: The images are noisy and sometimes contain only partial views of the bird, such as the head or wings.
- **Labels**: The target behaviors to classify include:
  - Feeding
  - Nesting
  - Preening
  - Resting
  - Foraging

## Methodology

1. **Pose Estimation**:
   - Detect key points of the Snow Petrel using a pose estimation model. Each key point represents specific parts of the bird (e.g., beak, wings, tail).

2. **Spatio-Temporal Graph Construction**:
   - Key points from each frame are treated as nodes, and relationships between them (i.e., spatial relations within a frame and temporal relations across frames) form the edges. This results in a spatio-temporal graph.

3. **ST-GCN Model**:
   - The Spatio-Temporal Graph Convolutional Network (ST-GCN) processes the keypoints (nodes) and their relations (edges) over time. This model learns how keypoints evolve to recognize patterns and classify behaviors.

4. **Behavior Classification**:
   - The ST-GCN outputs a classification for each sequence of frames, identifying the behavior of the Snow Petrel.

## Key Technologies
- **Pose Estimation**: Used for detecting key points of the Snow Petrel.
- **Spatio-Temporal Graph Convolutional Networks (ST-GCN)**: Graph-based neural networks used to model spatial and temporal relations between keypoints.
- **Python**: For data processing and modeling.
- **PyTorch / TensorFlow**: Deep learning frameworks for building and training the models.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/snow-petrel-behavior-detection.git
cd snow-petrel-behavior-detection
```

Set up a Python virtual environment (optional but recommended):
```bash
python3 -m venv env
source env/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset and place it in the `data/` directory.

## Usage

1. **Pose Estimation**:
   - Run the pose estimation model to detect key points in each image:
     ```bash
     python run_pose_estimation.py --input data/ --output keypoints/
     ```

2. **Graph Construction**:
   - Construct the spatio-temporal graph from the keypoints:
     ```bash
     python create_graph.py --keypoints keypoints/ --output graphs/
     ```

3. **Training ST-GCN**:
   - Train the ST-GCN model using the spatio-temporal graph data:
     ```bash
     python train_stgcn.py --data graphs/ --labels labels.csv --output model/
     ```

4. **Inference**:
   - Run the trained model on new data to predict behaviors:
     ```bash
     python predict.py --model model/stgcn.pth --data new_graphs/ --output predictions.csv
     ```

## Results
The model achieved an accuracy of **XX%** on the test set for behavior classification. The behaviors predicted include feeding, nesting, preening, resting, and foraging.

## Future Work
- Improve the pose estimation accuracy by using a more refined model for noisy environments.
- Explore combining ST-GCN with other architectures like 3D CNN for enhanced temporal understanding.
- Incorporate more contextual environmental data for behavior prediction.

## References
- Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *AAAI Conference on Artificial Intelligence*. [Link](https://aaai.org)
- Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [Link](https://ieeexplore.ieee.org)
