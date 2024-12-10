# **Yoga Pose Detection and Accuracy Feedback**

## **Overview**

This project uses **MediaPipe Pose** and **TensorFlow** to:
1. **Detect yoga poses** in videos using pose estimation.
2. **Calculate pose accuracy** by comparing detected poses to a reference pose.
3. **Classify yoga poses** using a pre-trained classification model (e.g., MobileNetV2 fine-tuned on a yoga pose dataset).
4. **Display real-time pose accuracy** on the video and generate a graph of accuracy over time.

## **Project Structure**

```
/yoga_pose_detection
    ├── yoga_pose_model.h5              # Pre-trained Yoga Pose Classification Model
    ├── yoga_example.jpg                # Sample input image (for testing pose detection)
    ├── 3327959-hd_1920_1080_24fps.mp4  # Sample input video (for testing video processing)
    ├── yoga_pose_detection.ipynb      # Colab notebook for pose detection and analysis
    ├── README.md                       # Project documentation
```

## **Dependencies**

This project requires the following libraries:

- **OpenCV**: For video and image processing.
- **MediaPipe**: For real-time pose estimation.
- **NumPy**: For numerical operations like keypoint processing.
- **Matplotlib**: For visualizing the pose accuracy plot.
- **TensorFlow**: For using the pre-trained Yoga Pose classification model.

You can install the required libraries using the following commands:

```bash
!pip install mediapipe opencv-python-headless tensorflow matplotlib numpy
```

## **Usage**

### 1. **Upload Your Own Video**
You can upload your own yoga video for pose detection and accuracy calculation. Follow these steps:

1. Upload your video using the Colab file upload interface:

```python
from google.colab import files
uploaded = files.upload()
```

2. Update the path in the `video_path` variable in the notebook to the path of your uploaded video.

```python
video_path = '/path/to/your/video.mp4'  # Update with your uploaded video
```

### 2. **Run the Code in the Colab Notebook**
After setting up the environment, follow these steps in the **Colab notebook**:

- **Step 1**: Install required libraries (already included in the notebook).
- **Step 2**: Import necessary libraries.
- **Step 3**: Load and process the video.
- **Step 4**: Extract keypoints and calculate pose accuracy.
- **Step 5**: Display pose accuracy and process each frame of the video.
- **Step 6**: Visualize the pose accuracy on a plot.

### 3. **Pose Classification**
If you have a pre-trained model (`yoga_pose_model.h5`), place it in the directory and update the model path in the notebook. The model is used to classify yoga poses detected in the frames.

```python
pose_classification_model = tf.keras.models.load_model('/path/to/your/yoga_pose_model.h5')
```

### 4. **Pose Accuracy Feedback**
The system will compare the pose in each frame to the **first detected pose** (reference pose). The calculated pose accuracy will be displayed in the form of a percentage.

### 5. **Graph Visualization**
At the end of the video processing, a graph showing the **frame-wise pose accuracy** will be displayed.

## **Contributing**

Feel free to fork the repository and contribute. If you have improvements or bug fixes, please submit a pull request.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
