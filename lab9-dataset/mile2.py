import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv('ACCELERATION.csv')
time = df['timestamp'].values
true_accel = df['acceleration'].values
noisy_accel = df['noisyacceleration'].values

dt = np.diff(time)
dt = np.append(dt, dt[-1])

true_velocity = np.cumsum(true_accel * dt)
noisy_velocity = np.cumsum(noisy_accel * dt)
true_distance = np.cumsum(true_velocity * dt)
noisy_distance = np.cumsum(noisy_velocity * dt)

plt.figure()
plt.plot(time, true_accel, 'b-', label='True Acceleration')
plt.plot(time, noisy_accel, 'r.', label='Noisy Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.savefig('acceleration_comparison.png')
plt.show()

plt.figure()
plt.plot(time, true_velocity, 'b-', label='True Velocity')
plt.plot(time, noisy_velocity, 'r-', label='Noisy Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.savefig('velocity_comparison.png')
plt.show()

plt.figure()
plt.plot(time, true_distance, 'b-', label='True Distance')
plt.plot(time, noisy_distance, 'r-', label='Noisy Distance')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.legend()
plt.savefig('distance_comparison.png')
plt.show()

print(f"Final true distance: {true_distance[-1]:.4f} m")
print(f"Final noisy distance: {noisy_distance[-1]:.4f} m")
print(f"Difference: {noisy_distance[-1] - true_distance[-1]:.4f} m")

walking_df = pd.read_csv('WALKING.csv')
accel_x = walking_df['accel_x'].values
accel_y = walking_df['accel_y'].values
accel_z = walking_df['accel_z'].values
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

window = 15
smoothed = np.convolve(accel_mag, np.ones(window)/window, mode='same')

peaks, _ = find_peaks(smoothed, height=12.0, distance=50)

plt.figure()
plt.plot(walking_df['timestamp'], accel_mag, alpha=0.5, label='Raw')
plt.plot(walking_df['timestamp'], smoothed, label='Smoothed')
plt.plot(walking_df['timestamp'].values[peaks], smoothed[peaks], 'ro', label='Steps')
plt.xlabel('Time (ns)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.savefig('step_detection_walking.png')
plt.show()

print(f"Number of steps detected: {len(peaks)}")

with open("TURNING.csv", 'r') as f:
    lines = f.readlines()
with open("TURNING_cleaned.csv", 'w') as f:
    for line in lines:
        f.write(line.rstrip(',\n') + '\n')

data = pd.read_csv("TURNING_cleaned.csv")

window = 51
data['gyro_smooth'] = np.convolve(data['gyro_z'], np.ones(window)/window, mode='same')

time_sec = (data['timestamp'] - data['timestamp'].iloc[0]) / 1e9
dt = np.diff(time_sec)
dt = np.insert(dt, 0, 0)
data['angle'] = np.degrees(np.cumsum(data['gyro_smooth'] * dt))

turns = []
for i in range(0, len(data) - 100, 50):
    for j in range(i + 100, min(i + 2000, len(data))):
        delta = data['angle'].iloc[j] - data['angle'].iloc[i]
        if abs(delta) >= 85 and abs(delta) <= 95:
            overlap = False
            for t in turns:
                if abs(i - t['start']) < 400 or abs(j - t['end']) < 400:
                    overlap = True
                    break
            if not overlap:
                turns.append({'angle': delta, 'start': i, 'end': j})
                break

plt.figure()
plt.plot(time_sec, data['angle'], label='Rotation')
plt.xlabel('Time (s)')
plt.ylabel('Degrees')
plt.legend()
plt.savefig('turn_detection.png')
plt.show()

print(f"\nDetected {len(turns)} turns:")
for i, t in enumerate(turns):
    print(f"Turn {i+1}: {t['angle']:.2f}°")

print("\n" + "="*40)
print("PART 4: Walking Trajectory")
print("="*40)

with open("WALKING_AND_TURNING.csv", 'r') as f:
    lines = f.readlines()
with open("WALKING_AND_TURNING_cleaned.csv", 'w') as f:
    for line in lines:
        f.write(line.rstrip(',\n') + '\n')

walk = pd.read_csv("WALKING_AND_TURNING_cleaned.csv")

accel_mag = np.sqrt(walk['accel_x']**2 + walk['accel_y']**2 + walk['accel_z']**2)
smoothed = np.convolve(accel_mag, np.ones(15)/15, mode='same')
step_peaks, _ = find_peaks(smoothed, height=12.0, distance=50)
print(f"Detected {len(step_peaks)} steps")

walk['gyro_smooth'] = np.convolve(walk['gyro_z'], np.ones(51)/51, mode='same')
walk['gyro_smooth'] = walk['gyro_smooth'] - walk['gyro_smooth'].mean()

time_sec = (walk['timestamp'] - walk['timestamp'].iloc[0]) / 1e9
dt = np.diff(time_sec)
dt = np.insert(dt, 0, 0)
walk['angle'] = np.degrees(np.cumsum(walk['gyro_smooth'] * dt))

detected_turns = []
for i in range(0, len(walk) - 100, 50):
    for j in range(i + 100, min(i + 2000, len(walk))):
        delta = walk['angle'].iloc[j] - walk['angle'].iloc[i]
        rounded = round(delta / 45) * 45
        if abs(delta - rounded) <= 10 and abs(rounded) >= 45:
            overlap = False
            for t in detected_turns:
                if abs(i - t['start']) < 400 or abs(j - t['end']) < 400:
                    overlap = True
                    break
            if not overlap:
                detected_turns.append({'rounded': rounded, 'start': i, 'end': j})
                break

print(f"Detected {len(detected_turns)} turns")

x = [0]
y = [0]
heading = 90

events = sorted(list(step_peaks) + [t['end'] for t in detected_turns])
turn_dict = {t['end']: t['rounded'] for t in detected_turns}

for idx in events:
    if idx in turn_dict:
        heading -= turn_dict[idx]
        if heading > 180: heading -= 360
        if heading < -180: heading += 360
    else:
        x.append(x[-1] + np.cos(np.radians(heading)))
        y.append(y[-1] + np.sin(np.radians(heading)))

plt.figure()
plt.plot(x, y, 'b-o', markersize=3)
plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.legend()
plt.savefig('walking_trajectory.png')
plt.show()

print(f"Final position: ({x[-1]:.2f}, {y[-1]:.2f}) m")
print(f"Total steps: {len(x) - 1}")
