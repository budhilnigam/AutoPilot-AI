"""
AutoPilot AI - Main Entry Point

Multi-Agent AI SRE System for Indian Startups
"""

import logging
import os
import sys
from typing import Dict, Any

from api.routes import AutoPilotAPI
from models.core_models import MetricData, MetricType
from services.scheduler import SchedulerService
from datetime import datetime


# Configure logging
def setup_logging():
    """Configure logging for AutoPilot AI"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('autopilot_ai.log')
        ]
    )
    
    # Reduce boto3 logging verbosity
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def main():
    """Main entry point for AutoPilot AI"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("AutoPilot AI - Multi-Agent AI SRE System")
    logger.info("=" * 60)
    
    try:
        # Initialize API
        api = AutoPilotAPI()

        # Start Background Scheduler
        scheduler = SchedulerService(api)
        scheduler.start()
        
        # Example usage
        print("\n" + "=" * 60)
        print("AutoPilot AI - Ready")
        print("=" * 60)
        print("\nExample Usage:")
        print("-" * 60)
        
        # Example 1: Simple query
        print("\n1. Natural Language Query:")
        result = api.query("What are my current AWS costs and how can I optimize them?")
        print(f"   Status: {result.get('status')}")
        print(f"   Agents Invoked: {', '.join([a.value for a in result.get('agents_invoked', [])])}")
        print(f"   Insights: {len(result.get('insights', []))} insights generated")
        if result.get('insights'):
            print(f"   Sample Insight: {result['insights'][0].get('summary', '')}")
        
        # Example 2: Metric analysis
        print("\n2. Metric Analysis:")
        metrics = [
            {
                'metric_name': 'CPUUtilization',
                'metric_type': 'cpu',
                'value': 85.5,
                'unit': 'Percent',
                'timestamp': datetime.utcnow().isoformat(),
                'dimensions': {'InstanceId': 'i-1234567890abcdef0'}
            },
            {
                'metric_name': 'MemoryUtilization',
                'metric_type': 'memory',
                'value': 72.3,
                'unit': 'Percent',
                'timestamp': datetime.utcnow().isoformat(),
                'dimensions': {'InstanceId': 'i-1234567890abcdef0'}
            }
        ]
        
        result = api.analyze_metrics(
            metrics=metrics,
            description="EC2 instance performance metrics"
        )
        print(f"   Status: {result.get('status')}")
        print(f"   Metrics Analyzed: 2")
        
        # Example 3: Configuration analysis
        print("\n3. Configuration Analysis:")
        dockerfile = """
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
        
        result = api.analyze_configuration(
            config_type='dockerfile',
            config_content=dockerfile,
            metadata={'service': 'api-server'}
        )
        print(f"   Status: {result.get('status')}")
        print(f"   Configuration Type: dockerfile")
        
        # Example 4: Tool generation
        print("\n4. Tool Generation:")
        result = api.generate_tool(
            tool_type='iam_policy',
            tool_spec={
                'purpose': 'S3 read-only access',
                'permissions': ['s3:GetObject', 's3:ListBucket']
            }
        )
        print(f"   Status: {result.get('status')}")
        print(f"   Tool Type: {result.get('tool_type', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("AutoPilot AI initialization complete!")
        print("=" * 60)
        print("\nThe system is ready to process queries.")
        print("All agents are registered and operational.")
        print("\nAvailable Agents:")
        print("  - Planner Agent (Orchestrator)")
        print("  - Observability Agent (Metrics & Anomalies)")
        print("  - Infrastructure Agent (Docker & ECS)")
        print("  - Database Agent (Query Optimization)")
        print("  - Cost Agent (Cost Optimization in INR)")
        print("  - CI/CD Agent (Build Performance)")
        print("\n" + "=" * 60)
        
        return api
        
    except Exception as e:
        logger.error(f"Failed to initialize AutoPilot AI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    api = main()
    
    # Interactive mode
    print("\nEntering interactive mode...")
    print("Type 'exit' to quit, or enter a query:\n")
    
    while True:
        try:
            user_input = input("autopilot> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Stopping scheduler...")
                scheduler.stop()
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\nProcessing query...\n")
            result = api.query(user_input)
            
            print(f"Status: {result.get('status')}")
            print(f"Task ID: {result.get('task_id')}")
            print(f"Execution Time: {result.get('execution_time_ms', 0):.1f}ms")
            print(f"\nAgents Invoked: {', '.join([a.value for a in result.get('agents_invoked', [])])}")
            
            insights = result.get('insights', [])
            if insights:
                print(f"\nInsights ({len(insights)}):")
                for i, insight in enumerate(insights, 1):
                    print(f"\n{i}. {insight.get('summary', '')}")
                    print(f"   Impact: {insight.get('business_impact', '')}")
                    print(f"   Severity: {insight.get('severity', 'N/A')}")
                    if insight.get('cost_impact_inr'):
                        print(f"   Cost Impact: ₹{insight.get('cost_impact_inr', 0):.2f} INR/month")
            
            recommendations = result.get('recommendations', [])
            if recommendations:
                print(f"\nRecommendations ({len(recommendations)}):")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"{i}. {rec}")
            
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            logger.error(f"Query processing error: {e}", exc_info=True)
