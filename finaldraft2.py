import cv2
from ultralytics import YOLO
import threading
import time
import requests
import json

# Load YOLO model
model = YOLO("yolo11n.pt")
camera_streams = {
    "road1": "http://192.168.29.181/stream",
    "road2": "http://192.168.29.105/stream",
    "road3": "http://192.168.29.115/stream"
}

# Define vehicle classes 
vehicle_classes = ["car", "truck", "bus", "motorbike"]

# Shared variables
vehicle_counts = {road: 0 for road in camera_streams}
traffic_status = {road: "Red Light" for road in camera_streams}
countdown_values = {road: 0 for road in camera_streams}
lock = threading.Lock()

# Time configurations
time_per_vehicle = 5  # Seconds per vehicle for green light
yellow_light_duration = 3  # Seconds for yellow light
default_green_time = 5  # Default green time for low traffic
frame_skip = 5  # Skip frames for optimization

# ESP32 configuration
esp32_url = "http://192.168.29.240/update-lights"  # Replace with your ESP32's IP address
terminate = False

def send_data_to_esp32():
    while not terminate:
        with lock:
            payload = {
                "road1": traffic_status["road1"],
                "road2": traffic_status["road2"],
                "road3": traffic_status["road3"],
            }
        try:
            response = requests.post(esp32_url, data=payload, timeout=2)
            if response.status_code != 200:
                print(f"Error sending data to ESP32: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data to ESP32: {e}")
        time.sleep(1)

def process_stream(stream_url, road_key):
    cap = cv2.VideoCapture(stream_url)
    frame_count = 0
    
    while not terminate:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read from stream {road_key}")
            time.sleep(1)
            continue
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames for efficiency
        
        resized_frame = cv2.resize(frame, (640, 480))
        results = model.predict(resized_frame)
        frame_vehicle_count = sum(1 for result in results for box in result.boxes if model.names[int(box.cls[0])] in vehicle_classes)
        
        with lock:
            vehicle_counts[road_key] = frame_vehicle_count
            status = traffic_status[road_key]
            countdown = countdown_values[road_key]
        
        cv2.putText(frame, f'Vehicles: {frame_vehicle_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Status: {status}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Countdown: {int(countdown)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow(road_key, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def traffic_light_control():
    roads = list(camera_streams.keys())
    while not terminate:
        with lock:
            counts = vehicle_counts.copy()
        total_vehicles = sum(counts.values())
        if total_vehicles == 0:
            with lock:
                for road in roads:
                    traffic_status[road] = "Red Light"
                    countdown_values[road] = 0
                traffic_status[roads[0]] = "Green Light"
                countdown_values[roads[0]] = default_green_time
            time.sleep(default_green_time)
            continue
        
        with lock:
            green_times = {road: max(default_green_time, counts[road] * time_per_vehicle) for road in roads}
        
        for road in roads:
            if counts[road] == 0:
                continue
            
            with lock:
                for r in roads:
                    traffic_status[r] = "Waiting" if counts[r] > 0 and r != road else "Red Light"
                    countdown_values[r] = green_times[road] + yellow_light_duration if r != road else 0
            
            with lock:
                traffic_status[road] = "Green Light"
                countdown_values[road] = green_times[road]
            
            start_time = time.time()
            while time.time() - start_time < green_times[road]:
                with lock:
                    countdown_values[road] = max(0, green_times[road] - (time.time() - start_time))
                time.sleep(1)
            
            with lock:
                traffic_status[road] = "Yellow Light"
                countdown_values[road] = yellow_light_duration
            
            start_time = time.time()
            while time.time() - start_time < yellow_light_duration:
                with lock:
                    countdown_values[road] = max(0, yellow_light_duration - (time.time() - start_time))
                time.sleep(1)
            
            with lock:
                countdown_values[road] = 0

if __name__ == "__main__":
    camera_threads = [threading.Thread(target=process_stream, args=(stream, road), daemon=True) for road, stream in camera_streams.items()]
    for thread in camera_threads:
        thread.start()
    
    control_thread = threading.Thread(target=traffic_light_control, daemon=True)
    control_thread.start()
    
    esp32_thread = threading.Thread(target=send_data_to_esp32, daemon=True)
    esp32_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        terminate = True
    
    for thread in camera_threads:
        thread.join()
    control_thread.join()
    esp32_thread.join()