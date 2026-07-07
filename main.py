import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo26n.pt')
cap = cv2.VideoCapture('your_video.mp4')

print("Opened:", cap.isOpened())
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame size:", frame_width, "x", frame_height)

cart_cutoff_y = int(frame_height * 0.55)
zone_left_x   = int(frame_width * 0.30)
zone_right_x  = int(frame_width * 0.70)

def draw_zone_lines(frame):
    cv2.line(frame, (0, cart_cutoff_y), (frame_width, cart_cutoff_y), (0, 255, 255), 2)
    cv2.line(frame, (zone_left_x, 0), (zone_left_x, cart_cutoff_y), (0, 255, 255), 2)
    cv2.line(frame, (zone_right_x, 0), (zone_right_x, cart_cutoff_y), (0, 255, 255), 2)
    return frame

def get_zone(center_x, center_y):
    if center_y >= cart_cutoff_y:
        return "CART"
    if center_x < zone_left_x:
        return "LEFT"
    elif center_x > zone_right_x:
        return "RIGHT"
    else:
        return "CENTER"

def get_zone_color(zone):
    colors = {
        "LEFT": (255, 100, 0),
        "CENTER": (0, 255, 0),
        "RIGHT": (0, 100, 255),
        "CART": (128, 128, 128)
    }
    return colors.get(zone, (0, 255, 0))

frame_num = 0
zone_visitors = {"LEFT": set(), "CENTER": set(), "RIGHT": set(), "CART": set()}
peak_occupancy = 0
occupancy_over_time = []
track_history = {}
fps = cap.get(cv2.CAP_PROP_FPS)

entry_count = 0
exit_count = 0
prev_zone_state = {}

heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
HEATMAP_RADIUS = 60
HEATMAP_BLEND = 0.25

output_path = 'output_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stopped at frame:", frame_num, "- ret was False")
        break

    results = model.track(frame, persist=True, tracker='custom_tracker.yaml', classes=[0], conf=0.3, verbose=False)
    r = results[0]
    annotated = frame.copy()
    current_occupancy = 0
    avg_live_dwell = 0
    live_health_score = 0

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            temp = np.zeros_like(heatmap)
            cv2.circle(temp, (int(center_x), int(center_y)), HEATMAP_RADIUS, 1, -1)
            heatmap += temp

            if track_id not in track_history:
                track_history[track_id] = {"first_frame": frame_num, "last_frame": frame_num}
            else:
                track_history[track_id]["last_frame"] = frame_num

            dwell_frames = track_history[track_id]["last_frame"] - track_history[track_id]["first_frame"]
            dwell_seconds = dwell_frames / fps

            zone = get_zone(center_x, center_y)
            zone_visitors[zone].add(track_id)
            box_color = get_zone_color(zone)

            if track_id not in prev_zone_state:
                prev_zone_state[track_id] = frame_num
                entry_count += 1

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(annotated, f'ID:{track_id} | {zone} | {dwell_seconds:.1f}s', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            cv2.circle(annotated, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        current_occupancy = len(track_ids)
        if current_occupancy > peak_occupancy:
            peak_occupancy = current_occupancy

        currently_visible = set(track_ids)
        for tid in list(prev_zone_state.keys()):
            if tid not in currently_visible and prev_zone_state[tid] != "EXITED":
                exit_count += 1
                prev_zone_state[tid] = "EXITED"

        live_dwells = []
        for tid in track_ids:
            info = track_history[tid]
            d = (info["last_frame"] - info["first_frame"]) / fps
            live_dwells.append(d)
        avg_live_dwell = sum(live_dwells) / len(live_dwells) if live_dwells else 0

        live_occupancy_score = min(100, (peak_occupancy / 5) * 100)
        live_dwell_score = min(100, (avg_live_dwell / 3) * 100)

        live_zone_counts = [len(zone_visitors["LEFT"]), len(zone_visitors["CENTER"]), len(zone_visitors["RIGHT"])]
        live_zone_avg = sum(live_zone_counts) / 3
        live_zone_std = (sum((x - live_zone_avg) ** 2 for x in live_zone_counts) / 3) ** 0.5
        live_balance_score = max(0, 100 - (live_zone_std / live_zone_avg * 100)) if live_zone_avg > 0 else 0

        live_health_score = (live_occupancy_score * 0.5) + (live_dwell_score * 0.3) + (live_balance_score * 0.2)

    heatmap_display = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_display = np.sqrt(heatmap_display)
    heatmap_normalized = cv2.normalize(heatmap_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    annotated = cv2.addWeighted(annotated, 1 - HEATMAP_BLEND, heatmap_color, HEATMAP_BLEND, 0)

    annotated = draw_zone_lines(annotated)

    zone_counts_now = {z: len(ids) for z, ids in zone_visitors.items() if z != "CART"}
    top_zone = max(zone_counts_now, key=zone_counts_now.get) if any(zone_counts_now.values()) else "N/A"

    cv2.rectangle(annotated, (0, 0), (340, 170), (0, 0, 0), -1)
    cv2.putText(annotated, f'Occupancy: {current_occupancy}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, f'Peak: {peak_occupancy}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, f'L:{len(zone_visitors["LEFT"])} C:{len(zone_visitors["CENTER"])} R:{len(zone_visitors["RIGHT"])}',
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, f'Avg dwell: {avg_live_dwell:.1f}s | Top zone: {top_zone}',
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(annotated, f'Health Score: {live_health_score:.0f}/100',
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated, f'Entries: {entry_count} | Exits: {exit_count}',
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    cv2.imshow("Test", annotated)
    out.write(annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1
    occupancy_over_time.append(current_occupancy)

cap.release()
out.release()
print("Saved annotated video to:", output_path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

time_axis = [i / fps for i in range(len(occupancy_over_time))]
plt.figure(figsize=(10, 4))
plt.plot(time_axis, occupancy_over_time, color='#2E75B6', linewidth=1.5)
plt.fill_between(time_axis, occupancy_over_time, alpha=0.2, color='#2E75B6')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of people visible')
plt.title('Store Occupancy Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('occupancy_over_time.png', dpi=150)
plt.close()
print("Saved occupancy graph to: occupancy_over_time.png")

final_heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
final_heatmap = np.sqrt(final_heatmap)
final_heatmap = cv2.normalize(final_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
final_heatmap_color = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)
cv2.imwrite('zone_heatmap.png', final_heatmap_color)
print("Saved heatmap image to: zone_heatmap.png")

print("Peak occupancy:", peak_occupancy)
print("Unique visitors per zone:")
for zone, ids in zone_visitors.items():
    print(f"  {zone}: {len(ids)} unique people")

print("\nDwell time per tracked person:")
dwell_results = []
for tid, info in track_history.items():
    dwell_s = (info["last_frame"] - info["first_frame"]) / fps
    dwell_results.append((tid, dwell_s))

dwell_results.sort(key=lambda x: x[1], reverse=True)
for tid, dwell_s in dwell_results:
    print(f"  ID {tid}: {dwell_s:.1f} seconds")

avg_dwell = 0
if dwell_results:
    avg_dwell = sum(d for _, d in dwell_results) / len(dwell_results)
    print(f"\nAverage dwell time: {avg_dwell:.1f} seconds")

final_zone_counts = {z: len(ids) for z, ids in zone_visitors.items() if z != "CART"}
final_top_zone = max(final_zone_counts, key=final_zone_counts.get) if any(final_zone_counts.values()) else "N/A"
print(f"\nMost popular zone overall: {final_top_zone}")

zone_ranking = sorted(final_zone_counts.items(), key=lambda x: x[1], reverse=True)
ranking_text = " > ".join([f"{z}({c})" for z, c in zone_ranking])
print(f"Zone ranking (most to least popular): {ranking_text}")

print(f"\nEntry events: {entry_count}")
print(f"Exit events: {exit_count}")

# --- Business Health Score ---
occupancy_score = min(100, (peak_occupancy / 5) * 100)
dwell_score = min(100, (avg_dwell / 3) * 100) if dwell_results else 0

zone_counts_list = [len(zone_visitors["LEFT"]), len(zone_visitors["CENTER"]), len(zone_visitors["RIGHT"])]
zone_avg = sum(zone_counts_list) / 3
zone_std = (sum((x - zone_avg) ** 2 for x in zone_counts_list) / 3) ** 0.5
balance_score = max(0, 100 - (zone_std / zone_avg * 100)) if zone_avg > 0 else 0

health_score = (occupancy_score * 0.5) + (dwell_score * 0.3) + (balance_score * 0.2)

print(f"\n--- BUSINESS HEALTH SCORE ---")
print(f"Occupancy score: {occupancy_score:.0f}/100")
print(f"Dwell score: {dwell_score:.0f}/100")
print(f"Zone balance score: {balance_score:.0f}/100")
print(f"OVERALL HEALTH SCORE: {health_score:.0f}/100")

cv2.destroyAllWindows()