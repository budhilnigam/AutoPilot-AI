
import sys
import os
import logging
import time

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from services.notification_service import NotificationService
from services.scheduler import SchedulerService
from api.routes import AutoPilotAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")

def test_integration():
    """Integration test for Scheduler and Notification services"""
    logger.info("Testing Notification Service instantiation...")
    notifier = NotificationService(region_name="us-east-1", sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic")
    assert notifier is not None
    logger.info("Notification Service instantiated.")

    logger.info("Testing Scheduler Service instantiation...")
    # Mock API to avoid full initialization overhead
    class MockAPI:
        def query(self, *args, **kwargs):
            return {'status': 'SUCCESS', 'insights': []}
            
    scheduler = SchedulerService(api=MockAPI())
    assert scheduler is not None
    logger.info("Scheduler Service instantiated.")
    
    logger.info("Starting and stopping scheduler...")
    scheduler.start()
    time.sleep(2)
    scheduler.stop()
    logger.info("Scheduler cycle complete.")

if __name__ == "__main__":
    try:
        test_integration()
        print("Integration Test Passed")
    except Exception as e:
        print(f"Integration Test Failed: {e}")
        sys.exit(1)
