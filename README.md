# **Yoga Pose Detection and Accuracy Feedback**

## **Overview**

This project leverages **MediaPipe Pose** and **TensorFlow** to:
1. **Detect yoga poses** in videos using pose estimation.
2. **Calculate pose accuracy** by comparing detected poses to a reference pose.
3. **Classify yoga poses** using a pre-trained classification model.
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

## **Approach**

The goal of this project is to process a yoga video, detect human poses, calculate the accuracy of the poses, and classify the yoga pose. The overall approach consists of the following steps:

1. **Pose Detection**:  
   We use **MediaPipe Pose** to detect body landmarks from video frames. These landmarks represent key body parts (e.g., shoulders, elbows, wrists, etc.).

2. **Pose Accuracy Calculation**:  
   Once the pose is detected, we compare the detected pose with a reference pose using the Euclidean distance between keypoints. The accuracy score is computed as a percentage.

3. **Pose Classification**:  
   We use a **pre-trained classification model** to identify the specific yoga pose (e.g., Warrior I, Tree Pose). The model is trained on a **Yoga-82 dataset** and classifies poses based on visual features from the frames.

4. **Feedback and Visualization**:  
   The calculated accuracy is displayed on each frame, and a final plot shows the accuracy of the pose detection over time.

## **Data Preprocessing**

### **Video Input**:  
The video input is processed frame-by-frame. We skip a specified number of frames for faster processing (controlled by `frame_skip`). The frames are converted to RGB format before feeding them into **MediaPipe Pose** for pose detection.

### **Pose Keypoints**:  
For each frame, the pose landmarks (keypoints) are detected and normalized. These keypoints represent various body parts, such as the shoulders, elbows, wrists, hips, knees, and ankles.

### **Pose Accuracy Calculation**:  
- The first frame's pose is used as a reference.  
- For each subsequent frame, the detected pose is compared to the reference pose by calculating the **Euclidean distance** between corresponding keypoints.
- The accuracy is normalized, giving a score between **0% to 100%**.

### **Pose Classification**:  
A **pre-trained MobileNetV2 model** (fine-tuned on the **Yoga-82 dataset**) classifies the pose in each frame. The model outputs a pose name (e.g., "Tree Pose") along with a confidence score.

## **Model Architecture**

The model used for **pose classification** is based on **MobileNetV2**, a lightweight CNN architecture. MobileNetV2 is designed for fast and efficient image classification tasks, especially on mobile and edge devices. The model was fine-tuned on a **Yoga-82 dataset** consisting of images of various yoga poses.

- **Input**:  
  - Image of size **224x224 pixels** (resized from the original frame).
  
- **Layers**:  
  - **Depthwise separable convolutions**: Efficient for mobile devices.  
  - **Global average pooling**: Reduces the output to a single vector.
  
- **Output**:  
  - The model outputs the **pose class** (e.g., Warrior I) and the **confidence score**.

### **Model Training** (Optional):
If you wish to train your own model, the following steps are required:
1. **Dataset**: Use the [Yoga-82 dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset) for training.
2. **Preprocessing**: Resize the images to **224x224 pixels** and normalize them.
3. **Fine-Tuning**: Fine-tune the MobileNetV2 model using transfer learning on the Yoga-82 dataset.

## **Results**

### **Pose Accuracy Feedback**:  
- After processing the video, the system computes the **pose accuracy** for each frame and displays it as feedback.
- The final accuracy is calculated by comparing the pose in each frame to the reference pose (the first detected pose).

### **Pose Classification**:  
- The pose detected in each frame is classified using the pre-trained model.
- The predicted pose name (e.g., Warrior I, Tree Pose) is displayed on the video.

### **Pose Accuracy Visualization**:  
- After the video is processed, a plot of the **frame-wise pose accuracy** is displayed, showing how the accuracy changes over time.

## **Next Steps**

1. **Improve Model**:
   - You can further fine-tune the **MobileNetV2 model** on a more extensive yoga pose dataset for better performance.
   - Use **data augmentation** (e.g., rotating, scaling images) to improve the robustness of the model.

2. **Real-time Pose Detection**:
   - Implement **real-time pose detection** using a webcam to analyze yoga poses in real-time.
   - Display live pose accuracy feedback during the practice.

3. **Mobile App Integration**:
   - Integrate the model into a mobile application, where users can track their yoga poses and get real-time feedback.

4. **Expand the Pose Library**:
   - Extend the **pose classification** model to support more yoga poses or other fitness exercises.

5. **User Feedback and Progress Tracking**:
   - Allow users to track their progress over time, including improvements in pose accuracy.

## **Contributing**

Feel free to fork the repository and contribute. If you have improvements or bug fixes, please submit a pull request.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
