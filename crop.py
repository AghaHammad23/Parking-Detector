import os
import cv2

output_dir = './data'
mask_path = './data/mask_1920_1080.png'

# Load the mask
mask = cv2.imread(mask_path, 0)  # Load in grayscale
if mask is None:
    raise FileNotFoundError(f"Error: The file at {mask_path} could not be read. Check the file path or file integrity.")

# Perform connected components analysis
analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis

# Extract bounding boxes for slots
slots = []
for i in range(1, totalLabels):
    x1 = values[i, cv2.CC_STAT_LEFT]
    y1 = values[i, cv2.CC_STAT_TOP]
    w = values[i, cv2.CC_STAT_WIDTH]
    h = values[i, cv2.CC_STAT_HEIGHT]
    slots.append([x1, y1, w, h])

# Process video
video_path = './samples/parking_1920_1080.mp4'
cap = cv2.VideoCapture(video_path)

frame_nmr = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
ret, frame = cap.read()
while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()
    if ret:
        for slot_nmr, slot in enumerate(slots):
            if slot_nmr in [132, 147, 164, 180, 344, 360, 377, 385, 341, 360, 179, 131, 106, 91, 61, 4, 89, 129,
                            161, 185, 201, 224, 271, 303, 319, 335, 351, 389, 29, 12, 32, 72, 281, 280, 157, 223, 26]:
                slot = frame[slot[1]:slot[1] + slot[3], slot[0]:slot[0] + slot[2], :]
                cv2.imwrite(os.path.join(output_dir, '{}_{}.jpg'.format(str(frame_nmr).zfill(8), str(slot_nmr).zfill(8))), slot)

        frame_nmr += 10
cap.release()
