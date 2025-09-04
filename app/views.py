from django.shortcuts import render
from django.http import JsonResponse
from .forms import ImageUploadForm
import cv2
import numpy as np
from PIL import Image
import math
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import base64
from io import BytesIO
import os
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

def home(request):
    return render(request, 'home.html')

pose_detector = mp.solutions.pose.Pose(static_image_mode=True)

# ----------------- NECKLACE FUNCTIONALITY ------------------

def necklace_tryon(request):
    necklace_images = os.listdir(os.path.join(settings.STATICFILES_DIRS[0], 'necklace'))
    return render(request, 'necklace.html', {'necklace_images': necklace_images})


@csrf_exempt
def validate_face_for_necklace(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided.'})

        image_data = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose_detector.process(img_rgb)
        if not results.pose_landmarks:
            return JsonResponse({'error': 'No shoulders detected.'})

        h, w = img.shape[:2]
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]

        left = np.array([int(left_shoulder.x * w), int(left_shoulder.y * h)])
        right = np.array([int(right_shoulder.x * w), int(right_shoulder.y * h)])

        shoulder_distance = np.linalg.norm(right - left)
        if shoulder_distance < 0.4 * w:
            return JsonResponse({'error': 'Please sit straight. Shoulders should face the camera.'})


        _, buffer = cv2.imencode('.png', img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        return JsonResponse({'image': encoded_img})

    return JsonResponse({'error': 'Invalid request.'})


@csrf_exempt
def apply_necklace(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        necklace_name = request.POST.get('necklace_name')

        if not image_file or not necklace_name:
            return JsonResponse({'error': 'Invalid request data.'})

        base_name, _ = os.path.splitext(necklace_name)
        cutout_path = os.path.join(settings.STATICFILES_DIRS[0], 'cutout_necklace', base_name + '.png')

        if not os.path.exists(cutout_path):
            return JsonResponse({'error': 'Necklace cutout not found.'})

        image_data = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({'error': 'Invalid image uploaded.'})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(img_rgb)    

        h, w = img.shape[:2]
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]

        left = np.array([int(left_shoulder.x * w), int(left_shoulder.y * h)])
        right = np.array([int(right_shoulder.x * w), int(right_shoulder.y * h)])

        (center_x, center_y), necklace_width, necklace_height = get_neck_region(results.pose_landmarks, w, h)

        necklace_img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
        if necklace_img is None:
            return JsonResponse({'error': 'Failed to load necklace cutout.'})

        necklace_resized = cv2.resize(necklace_img, (necklace_width, necklace_height))
        final_img = overlay_transparent(img, necklace_resized, center_x - necklace_width // 2, center_y - necklace_height // 2)

        _, buffer = cv2.imencode('.png', final_img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        return JsonResponse({'image': encoded_img})

    return JsonResponse({'error': 'Invalid request.'})


# ----------------- HELPER FUNCTIONS ------------------

def get_neck_region(landmarks, img_w, img_h):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    left = landmarks.landmark[LEFT_SHOULDER]
    right = landmarks.landmark[RIGHT_SHOULDER]

    left_pt = np.array([int(left.x * img_w), int(left.y * img_h)])
    right_pt = np.array([int(right.x * img_w), int(right.y * img_h)])

    center_x = int((left_pt[0] + right_pt[0]) / 2)
    center_y = int((left_pt[1] + right_pt[1]) / 2)

    shoulder_width = np.linalg.norm(right_pt - left_pt)
    width = int(shoulder_width * 0.75)
    height = int(width * 0.6)
    center_y -= int(height * 0.2)

    return (center_x, center_y), width, height


def overlay_transparent(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    overlay_x1, overlay_y1 = max(0, -x), max(0, -y)
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    bg_region = background[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay.shape[2] == 4:
        alpha = overlay_crop[:, :, 3:] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
        bg_region[:] = (alpha * overlay_rgb + (1 - alpha) * bg_region).astype(np.uint8)
    else:
        bg_region[:] = overlay_crop

    return background

#EARINGS-FUNCTIONALITY
ear_base_landmarks = {
    "left": [127, 234, 93],
    "right": [356, 454, 323]
}

def get_coords(index, landmarks, img_w, img_h):
    lm = landmarks.landmark[index]
    return _normalized_to_pixel_coordinates(lm.x, lm.y, img_w, img_h)

def detect_yaw_roll(landmarks, img_w, img_h):
    left_eye = get_coords(33, landmarks, img_w, img_h)
    right_eye = get_coords(263, landmarks, img_w, img_h)
    nose_tip = get_coords(1, landmarks, img_w, img_h)

    if None in (left_eye, right_eye, nose_tip):
        return None, None

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_distance = abs(right_eye[0] - left_eye[0])
    yaw_ratio = (nose_tip[0] - eye_center_x) / eye_distance
    yaw_angle = yaw_ratio * 60

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = math.degrees(math.atan2(dy, dx))

    return yaw_angle, roll_angle

def correct_roll(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated

def get_face_dimensions(landmarks, img_w, img_h):
    top = get_coords(10, landmarks, img_w, img_h)
    bottom = get_coords(152, landmarks, img_w, img_h)
    left = get_coords(234, landmarks, img_w, img_h)
    right = get_coords(454, landmarks, img_w, img_h)

    if None in (top, bottom, left, right):
        return None, None

    face_height = abs(bottom[1] - top[1])
    face_width = abs(right[0] - left[0])
    return face_width, face_height

def estimate_earlobe_position(side, landmarks, img_w, img_h, img_height):
    indices = ear_base_landmarks[side]
    sum_x = sum_y = 0
    count = 0
    for idx in indices:
        coords = get_coords(idx, landmarks, img_w, img_h)
        if coords:
            sum_x += coords[0]
            sum_y += coords[1]
            count += 1
    if count == 0:
        return None
    center_x = int(sum_x / count)
    center_y = int(sum_y / count)
    downward_offset = int(img_height * 0.05)
    return (center_x, center_y + downward_offset)

def overlay_earring(cv_img, pil_earring, position):
    bg_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    x, y = position
    bg_pil.paste(pil_earring, (x, y), pil_earring)
    return cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGBA2BGR)

def earrings(request):
    earring_images = os.listdir(os.path.join(settings.STATICFILES_DIRS[0], 'earrings'))
    return render(request, 'earrings.html', {'earring_images': earring_images})

def validate_face(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_bytes = uploaded_image.read()

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JsonResponse({'error': 'Invalid image format.'})

        image = cv2.resize(image, (400, int(image.shape[0] * 400 / image.shape[1])))
        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return JsonResponse({'error': 'No face detected.'})

        face_landmarks = results.multi_face_landmarks[0]
        yaw, roll = detect_yaw_roll(face_landmarks, img_w, img_h)

        if yaw is None or abs(yaw) > 3 or abs(roll) > 3:
            return JsonResponse({'error': 'Face is too turned or tilted. Please face the camera straight.'})

        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return JsonResponse({'image': img_base64})

    return JsonResponse({'error': 'Invalid request.'})

def apply_earring(request):
    if request.method == 'POST' and request.FILES.get('image') and request.POST.get('earring_name'):
        uploaded_image = request.FILES['image']
        earring_name = os.path.splitext(request.POST['earring_name'])[0]  # Remove extension

        image_bytes = uploaded_image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JsonResponse({'error': 'Invalid image format.'})

        image = cv2.resize(image, (400, int(image.shape[0] * 400 / image.shape[1])))
        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dynamically search cutout_er for matching base name with any extension
        cutout_dir = os.path.join(settings.STATICFILES_DIRS[0], 'cutout_er')
        matching_cutout = None
        for file in os.listdir(cutout_dir):
            if os.path.splitext(file)[0] == earring_name:
                matching_cutout = os.path.join(cutout_dir, file)
                break

        if not matching_cutout or not os.path.exists(matching_cutout):
            return JsonResponse({'error': 'Earring cutout not found.'})

        earring = Image.open(matching_cutout).convert("RGBA")

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return JsonResponse({'error': 'No face detected.'})

        face_landmarks = results.multi_face_landmarks[0]
        face_width, face_height = get_face_dimensions(face_landmarks, img_w, img_h)
        if face_width is None:
            return JsonResponse({'error': 'Could not compute face dimensions.'})

        scale_ratio = 0.18
        earring_height = int(face_height * scale_ratio)
        aspect_ratio = earring.width / earring.height
        earring_width = int(earring_height * aspect_ratio)
        earring_resized = earring.resize((earring_width, earring_height))

        for side in ['left', 'right']:
            earlobe = estimate_earlobe_position(side, face_landmarks, img_w, img_h, img_h)
            if earlobe:
                top_left = (earlobe[0] - earring_width // 2, earlobe[1])
                image = overlay_earring(image, earring_resized, top_left)

        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return JsonResponse({'image': img_base64})

    return JsonResponse({'error': 'Invalid request.'})

#RING-FUNCTIONALITY
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

def ring_tryon(request):
    ring_images = os.listdir(os.path.join(settings.STATICFILES_DIRS[0], 'rings'))
    return render(request, 'ring.html', {'ring_images': ring_images})

@csrf_exempt
def validate_hand(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided.'})

        image_data = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return JsonResponse({'error': 'No hand detected. Please try again.'})

        _, buffer = cv2.imencode('.png', img)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        return JsonResponse({'image': encoded_img})
    return JsonResponse({'error': 'Invalid request.'})


@csrf_exempt
def apply_ring(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        ring_name = request.POST.get('ring_name')
        selected_finger = request.POST.get('finger')

        if not image_file or not ring_name or not selected_finger:
            return JsonResponse({'error': 'Incomplete request data.'})

        base_name, _ = os.path.splitext(ring_name)
        ring_path = os.path.join(settings.STATICFILES_DIRS[0], 'cutout_ring', base_name + '.png')

        if not os.path.exists(ring_path):
            return JsonResponse({'error': 'Ring cutout not found.'})

        image_data = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return JsonResponse({'error': 'No hand detected.'})

        ring_pil = Image.open(ring_path).convert("RGBA")
        result_pil = Image.fromarray(img_rgb).convert("RGBA")
        h, w = img.shape[:2]

        fingers = {
            'index': {'mcp': 5, 'pip': 6},
            'middle': {'mcp': 9, 'pip': 10},
            'ring': {'mcp': 13, 'pip': 14},
            'little': {'mcp': 17, 'pip': 18}
        }

        if selected_finger not in fingers:
            return JsonResponse({'error': 'Invalid finger selected.'})

        for hand_landmarks in results.multi_hand_landmarks:
            joints = fingers[selected_finger]
            mcp = hand_landmarks.landmark[joints['mcp']]
            pip = hand_landmarks.landmark[joints['pip']]

            mcp_pos = (int(mcp.x * w), int(mcp.y * h))
            pip_pos = (int(pip.x * w), int(pip.y * h))

            dx = pip_pos[0] - mcp_pos[0]
            dy = pip_pos[1] - mcp_pos[1]
            length = np.sqrt(dx**2 + dy**2)
            angle_deg = np.degrees(np.arctan2(dy, dx)) - 90
            finger_width = length * 0.575

            scale_factor = finger_width / ring_pil.width
            new_size = (int(ring_pil.width * scale_factor), int(ring_pil.height * scale_factor))
            ring_resized = ring_pil.resize(new_size, Image.LANCZOS)
            ring_rotated = ring_resized.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

            ring_center_x = int(mcp_pos[0] + 0.6 * dx)
            ring_center_y = int(mcp_pos[1] + 0.6 * dy)
            paste_x = ring_center_x - ring_rotated.width // 2
            paste_y = ring_center_y - ring_rotated.height // 2

            result_pil.paste(ring_rotated, (paste_x, paste_y), ring_rotated)

        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return JsonResponse({'image': encoded_img})

    return JsonResponse({'error': 'Invalid request.'})
