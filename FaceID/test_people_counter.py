#!/usr/bin/env python3
"""
Test script for EdgeVision People Counter
"""

import cv2
import numpy as np
import time
from people_counter import PeopleCounter, AnalyticsData, PersonTrack, CONFIG
from datetime import datetime


def test_people_counter():
    """Test the people counting logic"""
    print("🧪 Testing People Counter...")

    # Create mock face analyzer
    class MockFaceAnalyzer:
        def get(self, frame):
            # Return mock face detections
            return [
                {
                    "bbox": [100, 100, 200, 200],
                    "embedding": np.random.rand(512),
                    "det_score": 0.9,
                }
            ]

    # Initialize people counter
    counter = PeopleCounter(MockFaceAnalyzer())

    # Test track updates
    mock_detections = [
        {"bbox": [100, 100, 200, 200], "identity": "TestPerson", "confidence": 0.9}
    ]

    frame_shape = (1080, 1920, 3)
    counter.update_tracks(mock_detections, frame_shape)

    print("✅ People Counter test passed")
    return True


def test_analytics_data():
    """Test analytics data structure"""
    print("🧪 Testing Analytics Data...")

    analytics = AnalyticsData()

    # Test data updates
    analytics.current_occupancy = 5
    analytics.peak_occupancy = 8
    analytics.average_dwell_time = 45.2

    print(f"   Current occupancy: {analytics.current_occupancy}")
    print(f"   Peak occupancy: {analytics.peak_occupancy}")
    print(f"   Average dwell time: {analytics.average_dwell_time}")

    print("✅ Analytics Data test passed")
    return True


def test_person_track():
    """Test person track dataclass"""
    print("🧪 Testing Person Track...")

    track = PersonTrack(track_id=1, identity="TestPerson", first_seen=datetime.now())

    # Add position
    track.positions.append((320, 240, time.time()))
    track.dwell_time = 30.5

    print(f"   Track ID: {track.track_id}")
    print(f"   Identity: {track.identity}")
    print(f"   Dwell time: {track.dwell_time}s")

    print("✅ Person Track test passed")
    return True


def test_config():
    """Test configuration values"""
    print("🧪 Testing Configuration...")

    print(f"   Flask port: {CONFIG['FLASK_PORT']}")
    print(f"   Dwell time threshold: {CONFIG['DWELL_TIME_THRESHOLD']}s")
    print(f"   Counting zone: {CONFIG['COUNTING_ZONE']}")

    # Validate counting zone
    zone = CONFIG["COUNTING_ZONE"]
    assert 0 <= zone["x"] <= 1, "Invalid zone x coordinate"
    assert 0 <= zone["y"] <= 1, "Invalid zone y coordinate"
    assert 0 <= zone["width"] <= 1, "Invalid zone width"
    assert 0 <= zone["height"] <= 1, "Invalid zone height"

    print("✅ Configuration test passed")
    return True


def main():
    """Run all tests"""
    print("🚀 EdgeVision People Counter - Test Suite")
    print("=" * 50)

    tests = [
        test_config,
        test_analytics_data,
        test_person_track,
        test_people_counter,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! People Counter is ready to run.")
        print(f"🌐 Start the system with: python people_counter.py")
        print(f"📊 Dashboard will be at: http://localhost:{CONFIG['FLASK_PORT']}")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

    return failed == 0


if __name__ == "__main__":
    main()
