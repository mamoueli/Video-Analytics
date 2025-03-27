# Load YOLOv8 for person detection
yolo_model = YOLO('yolov8n.pt')
deep_sort = DeepSort(max_age=30)

# Initialize InsightFace for face detection, age & gender
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load FairFace model for race classification
fairface_model = models.resnet34()
num_ftrs = fairface_model.fc.in_features
fairface_model.fc = torch.nn.Linear(num_ftrs, 18)
fairface_model.load_state_dict(torch.load("/content/res34_fair_align_multi_7_20190809.pt", map_location=torch.device('cpu')))
fairface_model.eval()

# Preprocessing for FairFace
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
race_classes = ["White", "Black", "Latino_Hispanic", "East_Asian", "Southeast_Asian", "Indian", "Middle_Eastern"]
gender_classes = ["Female", "Male"]
age_classes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

def predict_race(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = fairface_model(image)
    race_pred = output[0][:7]
    return race_classes[torch.argmax(race_pred).item()]

# Process video
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected_frames = [0, total_frames//3, 2*total_frames//3, total_frames-1]
frame_count = 0

for target_frame in selected_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    print(f"Processing Frame {target_frame}...")

    # YOLO Person Detection
    results = yolo_model(frame)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            label = yolo_model.names[class_id]
            if label == 'person':
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # DeepSORT Tracking
    tracked_objects = deep_sort.update_tracks(detections, frame=frame)
    persons_info = []

    for obj in tracked_objects:
        track_id = obj.track_id
        x1, y1, x2, y2 = map(int, obj.to_ltrb())
        person_crop = frame[y1:y2, x1:x2]

        # Crop the person's bounding box
        person_crop = frame[y1:y2, x1:x2]

# Skip empty or invalid crops
        if person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
            continue

# Run InsightFace detection
        faces = face_app.get(person_crop)

        faces = face_app.get(person_crop)
        for face in faces:
            bbox = face.bbox.astype(int)
            fx1, fy1, fx2, fy2 = bbox
            fx1, fy1, fx2, fy2 = fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1

            face_crop = frame[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            race = predict_race(face_pil)
            gender = "Male" if face.gender == 1 else "Female"
            age = int(face.age) if face.age else "?"

            demographics = {"Race": race, "Gender": gender, "Age": age}
            persons_info.append(f"Person {track_id}: {demographics}")

            # Draw annotations
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            label = f"{gender}, {race}, {age}"
           # Get size of text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

# Draw black rectangle as background
            cv2.rectangle(frame, (fx1, fy1 - text_h - 10), (fx1 + text_w + 4, fy1), (0, 0, 0), -1)

# Draw white text on top
            cv2.putText(frame, label, (fx1 + 2, fy1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print(f"Frame {target_frame} Analysis:")
    print("Persons:")
    for info in persons_info:
        print(info)
    print("-" * 50)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {target_frame}")
    plt.axis("off")
    plt.show()

cap.release()
cv2.destroyAllWindows()
