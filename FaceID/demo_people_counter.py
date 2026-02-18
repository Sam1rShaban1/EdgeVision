#!/usr/bin/env python3
"""
EdgeVision People Counting System - Demo Mode
Demonstrates the people counting capabilities without requiring camera hardware
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime, timedelta
from people_counter import PersonTrack, AnalyticsData, CONFIG


class DemoPeopleCounter:
    """Demo version of people counter for testing without camera"""

    def __init__(self):
        self.tracks = {}
        self.analytics = AnalyticsData()
        self.next_track_id = 1
        self.demo_time = 0

    def create_demo_frame(self):
        """Create a demo frame with simulated people"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 20, 30)  # Dark background

        # Add title
        cv2.putText(
            frame,
            "EdgeVision People Counter - Demo Mode",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Add counting zone
        zone = CONFIG["COUNTING_ZONE"]
        zone_x = int(zone["x"] * frame.shape[1])
        zone_y = int(zone["y"] * frame.shape[0])
        zone_w = int(zone["width"] * frame.shape[1])
        zone_h = int(zone["height"] * frame.shape[0])

        cv2.rectangle(
            frame,
            (zone_x, zone_y),
            (zone_x + zone_w, zone_y + zone_h),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Counting Zone",
            (zone_x, zone_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        # Simulate people movement
        self._simulate_people_movement(frame)

        # Add analytics overlay
        self._add_demo_analytics(frame)

        return frame

    def _simulate_people_movement(self, frame):
        """Simulate people moving through the scene"""
        self.demo_time += 0.1

        # Create periodic new tracks
        if int(self.demo_time) % 3 == 0 and len(self.tracks) < 5:
            new_track = PersonTrack(
                track_id=self.next_track_id,
                identity=f"DemoPerson_{self.next_track_id}",
                first_seen=datetime.now(),
            )

            # Start from left side
            start_x = 50
            start_y = 200 + (self.next_track_id * 50)
            new_track.positions.append((start_x, start_y, time.time()))

            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

            # Update analytics
            self.analytics.total_people_today += 1

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]

            # Move person across frame
            if track.positions:
                last_pos = track.positions[-1]
                new_x = last_pos[0] + 5  # Move right
                new_y = (
                    last_pos[1] + np.sin(self.demo_time + track_id) * 2
                )  # Slight wave motion

                # Remove if out of frame
                if new_x > frame.shape[1] - 50:
                    del self.tracks[track_id]
                    continue

                track.positions.append((new_x, new_y, time.time()))
                track.last_seen = datetime.now()
                track.dwell_time = (track.last_seen - track.first_seen).total_seconds()

                # Draw person
                color = (0, 255, 0) if track.identity != "Unknown" else (0, 0, 255)
                cv2.circle(frame, (int(new_x), int(new_y)), 15, color, -1)

                # Draw track info
                info_text = (
                    f"ID: {track.track_id} | {track.identity} | {track.dwell_time:.1f}s"
                )
                cv2.putText(
                    frame,
                    info_text,
                    (int(new_x) + 20, int(new_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

                # Draw trail
                for i in range(1, len(track.positions)):
                    pos1 = (
                        int(track.positions[i - 1][0]),
                        int(track.positions[i - 1][1]),
                    )
                    pos2 = (int(track.positions[i][0]), int(track.positions[i][1]))
                    cv2.line(frame, pos1, pos2, (0, 255, 0), 1)

    def _add_demo_analytics(self, frame):
        """Add demo analytics overlay"""
        # Update analytics
        self.analytics.current_occupancy = len(self.tracks)
        self.analytics.peak_occupancy = max(
            self.analytics.peak_occupancy, self.analytics.current_occupancy
        )

        if self.tracks:
            avg_dwell = np.mean([track.dwell_time for track in self.tracks.values()])
            self.analytics.average_dwell_time = avg_dwell

        # Count people in zone
        zone_people = 0
        zone = CONFIG["COUNTING_ZONE"]
        for track in self.tracks.values():
            if track.positions:
                pos = track.positions[-1]
                if (
                    zone["x"] <= pos[0] / 640 <= zone["x"] + zone["width"]
                    and zone["y"] <= pos[1] / 480 <= zone["y"] + zone["height"]
                ):
                    zone_people += 1

        self.analytics.people_in_zone = zone_people

        # Draw analytics info
        analytics_text = [
            f"Occupancy: {self.analytics.current_occupancy}",
            f"Total Today: {self.analytics.total_people_today}",
            f"Peak: {self.analytics.peak_occupancy}",
            f"Avg Dwell: {self.analytics.average_dwell_time:.1f}s",
            f"In Zone: {self.analytics.people_in_zone}",
            f"Demo Time: {self.demo_time:.1f}s",
        ]

        for i, text in enumerate(analytics_text):
            cv2.putText(
                frame,
                text,
                (10, 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )


def main():
    """Run demo people counting system"""
    print("🎯 EdgeVision People Counter - Demo Mode")
    print("=" * 50)
    print(
        "This demo shows the people counting capabilities without requiring camera hardware."
    )
    print("Press 'q' to quit the demo.")
    print("=" * 50)

    demo_counter = DemoPeopleCounter()

    while True:
        # Create demo frame
        frame = demo_counter.create_demo_frame()

        # Show frame
        cv2.imshow("EdgeVision People Counter - Demo", frame)

        # Check for quit
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

        # Control frame rate
        time.sleep(0.03)  # ~30 FPS

    cv2.destroyAllWindows()

    # Print final analytics
    print("\n📊 Demo Session Analytics:")
    print(f"   Total People: {demo_counter.analytics.total_people_today}")
    print(f"   Peak Occupancy: {demo_counter.analytics.peak_occupancy}")
    print(f"   Average Dwell Time: {demo_counter.analytics.average_dwell_time:.1f}s")
    print(f"   Session Duration: {demo_counter.demo_time:.1f}s")

    print("\n🎉 Demo complete! To run the full system:")
    print("   1. Install dependencies: pip install -r requirements_people_counter.txt")
    print("   2. Generate face database: python try.py")
    print("   3. Start people counter: python people_counter.py")
    print("   4. Access dashboard at: http://localhost:5001")


if __name__ == "__main__":
    main()
