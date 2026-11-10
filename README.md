# Virtual Jewellery Try-On Using MediaPipe and Computer Vision

An interactive, computer vision-based web application that enables users to virtually try on **rings, necklaces, and earrings** using their own photographs or camera-captured images.

Built with **Python, Django, MediaPipe, OpenCV, and Pillow**, the application combines anatomical landmark detection, geometric transformations, and image compositing to position virtual jewellery on the user's body.

## Project Overview

Traditional online jewellery shopping makes it difficult for customers to visualize how accessories will look when worn. This project addresses that limitation through an image-based virtual try-on system.

Rather than placing jewellery at fixed coordinates, the application uses computer vision techniques to detect anatomical landmarks and estimate suitable positions, orientations, and dimensions for jewellery overlays.

The system supports three categories:

- **Ring Try-On:** Finger landmark detection and ring alignment.
- **Necklace Try-On:** Shoulder landmark detection and necklace placement.
- **Earring Try-On:** Facial landmark detection and earring positioning.

## Key Features

### 1. Virtual Ring Try-On

Uses **MediaPipe Hands** to detect hand and finger landmarks.

- Detects finger positions from uploaded or captured images.
- Supports index, middle, ring, and little finger selection.
- Calculates jewellery placement using finger landmark coordinates.
- Applies resizing and rotation to align virtual rings with the selected finger.
- Overlays transparent ring images while maintaining visual proportions.

### 2. Virtual Necklace Try-On

Uses **MediaPipe Pose** to identify relevant body landmarks for necklace positioning.

- Detects shoulder landmarks to estimate body alignment.
- Validates whether the uploaded image is suitable for necklace placement.
- Estimates necklace size and position from anatomical reference points.
- Applies image scaling and positioning to create a realistic visual preview.
- Accommodates slight variations in head orientation.

### 3. Virtual Earring Try-On

Uses facial landmark detection to estimate appropriate earring positions.

- Identifies facial reference points for earring placement.
- Positions virtual earrings relative to the detected landmarks.
- Adjusts overlay dimensions and coordinates.
- Supports transparent jewellery images for natural-looking visualization.

## Technical Implementation

The project follows a modular image-processing workflow:

**1. Image Acquisition**

Users upload an existing image or capture one using their device camera.

**2. Landmark Detection**

MediaPipe processes the image to estimate relevant hand, facial, or body landmarks, depending on the selected jewellery category.

**3. Geometric Transformation**

The system calculates the required overlay position, scale, and rotation using detected landmark coordinates.

**4. Image Compositing**

OpenCV and Pillow support image manipulation and the placement of transparent jewellery overlays on the original image.

**5. Result Visualization**

The processed image is returned through the Django application, allowing the user to preview the selected jewellery.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic and image-processing workflows |
| Django | Web application framework and backend processing |
| MediaPipe | Hand, body pose, and facial landmark detection |
| OpenCV | Image processing and geometric transformations |
| Pillow (PIL) | Image manipulation and transparent overlays |
| HTML, CSS, JavaScript | Interactive frontend and user experience |

## Computer Vision Concepts Demonstrated

This project applies several fundamental computer vision concepts:

- **Landmark Detection:** Extracting anatomical reference points from input images.
- **Spatial Geometry:** Calculating positions, distances, and orientations from landmark coordinates.
- **Image Transformation:** Scaling and rotating jewellery assets to match the user's anatomy.
- **Image Compositing:** Combining transparent jewellery overlays with photographs.
- **Input Validation:** Identifying unsuitable images to improve placement reliability.

## Getting Started

### Prerequisites

- Python
- pip
- A webcam or sample images for testing

### Installation

Clone the repository:

```bash
git clone https://github.com/JhanModi/jewellery-try-on-mediapipe-.git
cd jewellery-try-on-mediapipe-
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run database migrations:

```bash
python manage.py migrate
```

Start the Django development server:

```bash
python manage.py runserver
```

Open the application at:

`http://127.0.0.1:8000/`

## Project Structure

```text
jewellery-try-on-mediapipe-/
├── app/                 # Application features and processing
├── myproject/           # Django project configuration
├── manage.py            # Django management utility
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Engineering Highlights

The central challenge of this project is translating raw anatomical landmark coordinates into visually meaningful jewellery placement.

The application addresses this through a category-specific processing approach:

- **Rings:** Placement follows finger geometry rather than fixed image positions.
- **Necklaces:** Shoulder landmarks guide alignment and proportional sizing.
- **Earrings:** Facial reference points guide positioning across different face images.

Separating these processing workflows also makes the application easier to maintain and extend with additional jewellery categories.

## Limitations and Future Improvements

The quality of the virtual try-on depends on landmark detection accuracy, lighting conditions, camera angles, and image resolution.

Potential improvements include:

- Real-time video-based jewellery tracking.
- Improved handling of occlusion and challenging camera angles.
- More realistic lighting and shadow adjustments.
- Automatic jewellery scaling based on estimated physical dimensions.
- GPU-assisted processing for more responsive performance.

## Applications

This project demonstrates how computer vision can improve product visualization in:

- Online jewellery retail.
- Virtual shopping experiences.
- Interactive product demonstrations.
- Augmented reality-inspired e-commerce applications.

## Author

**Jhan Modi**

[GitHub](https://github.com/JhanModi) | [LinkedIn](https://linkedin.com/in/jhan-modi-28aa05292)

---

**Project Focus:** Computer Vision • Landmark Detection • Image Processing • Geometric Transformations • Python • Django
