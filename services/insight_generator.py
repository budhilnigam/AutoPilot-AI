"""
Semantic Insight Generator

Transforms raw AWS metrics into business-focused insights with context, recommendations, and cost impact.
This is what makes AutoPilotAI different from traditional monitoring tools.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from services.currency_formatter import CurrencyFormatter, format_cost_savings, format_roi

logger = logging.getLogger(__name__)


class SemanticInsightGenerator:
    """
    Generate semantic insights from raw AWS data.
    
    Instead of: "CPU: 78%"
    We generate: "Your worker pool is undersized for Redis throughput. Scale to 12 workers. 
                  Cost: +₹2,400/mo, Saves: ₹18,300/mo, ROI: 662%"
    """
    
    def __init__(self):
        """Initialize the semantic insight generator."""
        self.currency = CurrencyFormatter()
        logger.info("Semantic Insight Generator initialized")
    
    def analyze_high_cpu(
        self,
        resource_type: str,
        resource_name: str,
        cpu_percent: float,
        additional_context: Dict[str, Any] = None
    ) -> Dict[str,Any]:
        """
        Analyze high CPU utilization with business context.
        
        Args:
            resource_type: Type of resource (ECS, EC2, Lambda)
            resource_name: Resource identifier
            cpu_percent: CPU utilization percentage
            additional_context: Additional metrics (memory, queue depth, etc.)
            
        Returns:
            Semantic insight with recommendations
        """
        additional_context = additional_context or {}
        
        insight = {
            "summary": f"High CPU utilization detected on {resource_type}: {resource_name}",
            "severity": "HIGH" if cpu_percent > 80 else "MEDIUM",
            "metrics": {
                "cpu_utilization": f"{cpu_percent}%",
                "threshold": "75%",
                "deviation": f"+{cpu_percent - 75}%"
            }
        }
        
        # Determine root cause based on resource type and context
        if resource_type == "ECS":
            current_tasks = additional_context.get("task_count", 2)
            queue_depth = additional_context.get("queue_depth", 0)
            
            # Calculate recommended task count
            # If CPU > 80%, we likely need more tasks
            target_cpu = 60  # Target utilization
            recommended_tasks = max(2, int(current_tasks * (cpu_percent / target_cpu)))
            
            # Cost calculation (approximate)
            # Assume Fargate 0.25 vCPU, 0.5 GB = ~$12/month = ₹1,000/month per task
            cost_per_task_usd = 12
            current_cost = current_tasks * cost_per_task_usd
            proposed_cost = recommended_tasks * cost_per_task_usd
            
            savings_info = format_cost_savings(current_cost, proposed_cost)
            
            insight.update({
                "root_cause": f"Worker pool ({current_tasks} tasks) undersized for current load",
                "business_impact": "Job processing delays, potential timeout failures, degraded user experience",
                "recommendation": {
                    "action": f"Scale ECS service from {current_tasks} to {recommended_tasks} tasks",
                    "implementation": "Update ECS service desired count or enable auto-scaling",
                    "risk": "Low - Horizontal scaling is safe",
                    "effort": "Low - 5 minutes via AWS Console or CLI"
                },
                "cost_impact": {
                    **savings_info,
                    "additional_savings": "Prevents over-provisioning of downstream services (Redis/RDS)"
                },
                "priority": 1,
                "urgency": "Implement within 24 hours"
            })
            
            if queue_depth > 1000:
                insight["additional_context"] = f"Queue backlog of {queue_depth} jobs detected - scaling is urgent"
        
        elif resource_type == "RDS":
            # Database high CPU - likely missing indexes or inefficient queries
            insight.update({
                "root_cause": "Database queries are CPU-intensive - likely missing indexes or unoptimized queries",
                "business_impact": "Slow query response times affecting API performance and user experience",
                "recommendation": {
                    "action": "Analyze slow query log and add missing indexes",
                    "implementation": "Use RDS Performance Insights to identify slow queries, then CREATE INDEX",
                    "risk": "Low - Indexes improve read performance",
                    "effort": "Medium - Requires query analysis and testing"
                },
                "next_steps": [
                    "Enable RDS Performance Insights if not already enabled",
                    "Query Top SQL by CPU consumption",
                    "Run EXPLAIN on top queries",
                    "Add appropriate indexes"
                ],
                "priority": 1
            })
        
        elif resource_type == "EC2":
            instance_type = additional_context.get("instance_type", "unknown")
            
            insight.update({
                "root_cause": f"EC2 instance {instance_type} at capacity - application load exceeds instance size",
                "business_impact": "Application slowdown, potential request timeouts",
                "recommendation": {
                    "action": "Upgrade instance type or add instances to Auto Scaling Group",
                    "implementation": "Consider moving to containers (ECS) for better cost optimization",
                    "risk": "Medium - Requires testing and potential downtime",
                    "effort": "Medium - 1-2 hours including testing"
                },
                "cost_optimization_tip": "Containerizing this workload could reduce costs by 40-60%",
                "priority": 2
            })
        
        return insight
    
    def analyze_underutilized_resource(
        self,
        resource_type: str,
        resource_name: str,
        utilization: Dict[str, float],
        instance_specs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze underutilized resources with cost savings recommendations.
        
        Args:
            resource_type: Type of resource (RDS, EC2, ECS)
            resource_name: Resource identifier
            utilization: Utilization metrics (cpu, memory, etc.)
            instance_specs: Instance specifications
            
        Returns:
            Semantic insight with cost savings
        """
        instance_specs = instance_specs or {}
        
        cpu = utilization.get("cpu", 0)
        memory = utilization.get("memory", 0)
        avg_utilization = (cpu + memory) / 2
        
        insight = {
            "summary": f"Underutilized resource detected: {resource_type} {resource_name}",
            "severity": "MEDIUM",
            "metrics": {
                "cpu_utilization": f"{cpu}%",
                "memory_utilization": f"{memory}%",
                "average_utilization": f"{avg_utilization}%",
                "optimal_range": "60-80%"
            }
        }
        
        if resource_type == "RDS":
            current_instance = instance_specs.get("instance_class", "db.r5.large")
            
            # Suggest downgrade if utilization < 30%
            if avg_utilization < 30:
                # Simplified downgrade logic
                downgrade_map = {
                    "db.r5.2xlarge": ("db.r5.xlarge", 500, 250),
                    "db.r5.xlarge": ("db.r5.large", 250, 140),
                    "db.r5.large": ("db.t3.large", 140, 70),
                    "db.t3.large": ("db.t3.medium", 70, 35),
                }
                
                if current_instance in downgrade_map:
                    recommended, current_cost_usd, new_cost_usd = downgrade_map[current_instance]
                    savings_info = format_cost_savings(current_cost_usd, new_cost_usd)
                    
                    insight.update({
                        "root_cause": f"Database instance {current_instance} is over-provisioned for current load",
                        "business_impact": f"Wasting {savings_info['savings']} on unused database capacity",
                        "recommendation": {
                            "action": f"Downgrade from {current_instance} to {recommended}",
                            "implementation": "Use AWS RDS modify-db-instance CLI or Console",
                            "risk": "Low - Current utilization provides comfortable headroom",
                            "effort": "Low - 15 minutes (requires brief downtime)",
                            "downtime": "~5-10 minutes during instance modification"
                        },
                        "cost_impact": {
                            **savings_info,
                            "annual_impact": self.currency.format_inr(savings_info['savings_inr'] * 12),
                            "confidence": "High - Based on 7-day average utilization"
                        },
                        "performance_impact": f"Minimal - {recommended} can handle {int(avg_utilization * 2)}% utilization",
                        "priority": 2
                    })
        
        elif resource_type == "EC2":
            current_type = instance_specs.get("instance_type", "t3.large")
            
            insight.update({
                "root_cause": f"EC2 instance {current_type} underutilized - running at {avg_utilization}%",
                "business_impact": "Paying for compute capacity you're not using",
                "recommendation": {
                    "action": f"Downgrade instance type or migrate to Fargate/ECS for auto-scaling",
                    "implementation": "Consider containerization for better resource utilization",
                    "risk": "Low if proper testing is done",
                    "effort": "Medium - Depends on application architecture"
                },
                "modernization_opportunity": "Moving to containers could reduce costs by 40-60%",
                "priority": 3
            })
        
        return insight
    
    def analyze_cost_anomaly(
        self,
        service_name: str,
        current_cost: float,
        baseline_cost: float,
        time_period: str = "month"
    ) -> Dict[str, Any]:
        """
        Analyze cost anomalies and spikes.
        
        Args:
            service_name: AWS service name
            current_cost: Current cost in USD
            baseline_cost: Baseline/expected cost in USD
            time_period: Time period for comparison
            
        Returns:
            Semantic insight about cost anomaly
        """
        increase_percent = ((current_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
        increase_amount_usd = current_cost - baseline_cost
        increase_amount_inr = self.currency.usd_to_inr(increase_amount_usd)
        
        insight = {
            "summary": f"Cost anomaly detected for {service_name}",
            "severity": "HIGH" if increase_percent > 30 else "MEDIUM",
            "metrics": {
                "current_cost": self.currency.format_usd_as_inr(current_cost),
                "baseline_cost": self.currency.format_usd_as_inr(baseline_cost),
                "increase": self.currency.format_inr(increase_amount_inr),
                "increase_percent": f"{increase_percent:.1f}%"
            },
            "root_cause": f"{service_name} costs increased by {increase_percent:.0f}% compared to baseline",
            "business_impact": f"Unexpected spend of {self.currency.format_inr(increase_amount_inr)} this {time_period}",
            "investigation_steps": [
                f"Check {service_name} usage metrics for unusual activity",
                "Review recent deployments or configuration changes",
                "Analyze resource utilization trends",
                "Check for inefficient resource usage patterns"
            ],
            "priority": 1 if increase_percent > 50 else 2
        }
        
        # Service-specific recommendations
        if service_name == "RDS":
            insight["possible_causes"] = [
                "Database instance size increased",
                "High I/O operations",
                "Increased backup storage",
                "Multi-AZ replication added"
            ]
        elif service_name == "EC2":
            insight["possible_causes"] = [
                "Auto-scaling triggered more instances",
                "Larger instance types launched",
                "Increased data transfer costs",
                "EBS volume expansion"
            ]
        elif service_name == "Lambda":
            insight["possible_causes"] = [
                "Increased invocation count",
                "Longer execution times",
                "Higher memory allocation",
                "Increased data processing"
            ]
        
        return insight
    
    def analyze_build_regression(
        self,
        build_name: str,
        current_duration: int,
        baseline_duration: int,
        commit_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze CI/CD build time regressions.
        
        Args:
            build_name: Build/workflow name
            current_duration: Current build duration in seconds
            baseline_duration: Baseline duration in seconds
            commit_info: Commit information
            
        Returns:
            Semantic insight about build regression
        """
        increase_percent = ((current_duration - baseline_duration) / baseline_duration * 100) if baseline_duration > 0 else 0
        increase_seconds = current_duration - baseline_duration
        
        insight = {
            "summary": f"Build time regression detected: {build_name}",
            "severity": "MEDIUM" if increase_percent > 20 else "LOW",
            "metrics": {
                "current_duration": f"{current_duration}s",
                "baseline_duration": f"{baseline_duration}s",
                "increase": f"+{increase_seconds}s",
                "increase_percent": f"{increase_percent:.1f}%"
            },
            "root_cause": f"Build duration increased by {increase_percent:.0f}% compared to baseline",
            "business_impact":"Slower deployment cycles, reduced developer productivity",
            "recommendation": {
                "action": "Investigate and optimize build process",
                "priority_checks": [
                    "Review recent dependency additions",
                    "Check for missing build caching",
                    "Analyze test execution time",
                    "Consider parallelizing build steps"
                ],
                "effort": "Medium - Requires build pipeline analysis"
            },
            "priority": 2 if increase_percent > 40 else 3
        }
        
        if commit_info:
            insight["attributed_commit"] = {
                "sha": commit_info.get("sha", "unknown")[:7],
                "author": commit_info.get("author", "unknown"),
                "message": commit_info.get("message", "unknown")
            }
            insight["recommendation"]["specific_action"] = f"Review changes in commit {commit_info.get('sha', '')[:7]}"
        
        # Common causes
        insight["common_causes"] = [
            "Heavy dependencies added (check package.json, requirements.txt)",
            "Missing Docker layer caching",
            "Test suite expanded without optimization",
            "Build runner performance degradation"
        ]
        
        return insight
    
    def generate_health_summary(
        self,
        service_statuses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate overall infrastructure health summary.
        
        Args:
            service_statuses: Dictionary of service names to their status/metrics
            
        Returns:
            Comprehensive health summary with priorities
        """
        total_services = len(service_statuses)
        healthy_count = sum(1 for s in service_statuses.values() if s.get("status") == "healthy")
        warnings = sum(1 for s in service_statuses.values() if s.get("status") == "warning")
        critical = sum(1 for s in service_statuses.values() if s.get("status") == "critical")
        
        overall_health = "HEALTHY" if critical == 0 and warnings <= 1 else "WARNING" if critical == 0 else "CRITICAL"
        
        summary = {
            "overall_health": overall_health,
            "summary": f"{healthy_count}/{total_services} services healthy",
            "metrics": {
                "total_services": total_services,
                "healthy": healthy_count,
                "warnings": warnings,
                "critical": critical,
                "health_score": int((healthy_count / total_services * 100)) if total_services > 0 else 0
            },
            "timestamp": datetime.utcnow().isoformat(),
            "priorities": []
        }
        
        # Identify top priorities
        issues = []
        for name, status in service_statuses.items():
            if status.get("status") != "healthy":
                issues.append({
                    "service": name,
                    "severity": status.get("status").upper(),
                    "issue": status.get("issue", "Unknown issue"),
                    "recommendation": status.get("recommendation", "Investigate immediately")
                })
        
        # Sort by severity
        severity_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2}
        issues.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        summary["priorities"] = issues[:5]  # Top 5 issues
        
        if critical > 0:
            summary["action_required"] = f"IMMEDIATE: {critical} critical issue(s) require attention"
        elif warnings > 0:
            summary["action_required"] = f"Review {warnings} warning(s) within 24 hours"
        else:
            summary["action_required"] = "No immediate action required - system healthy"
        
        return summary


# Global instance for convenience
_generator = SemanticInsightGenerator()


def generate_high_cpu_insight(resource_type: str, resource_name: str, cpu_percent: float, context: Dict = None) -> Dict:
    """Convenience function for high CPU analysis."""
    return _generator.analyze_high_cpu(resource_type, resource_name, cpu_percent, context)


def generate_underutilized_insight(resource_type: str, resource_name: str, utilization: Dict, specs: Dict = None) -> Dict:
    """Convenience function for underutilization analysis."""
    return _generator.analyze_underutilized_resource(resource_type, resource_name, utilization, specs)


def generate_cost_anomaly_insight(service: str, current: float, baseline: float) -> Dict:
    """Convenience function for cost anomaly analysis."""
    return _generator.analyze_cost_anomaly(service, current, baseline)


if __name__ == "__main__":
    # Test the generator
    print("=== Semantic Insight Generator Tests ===\n")
    
    generator = SemanticInsightGenerator()
    
    # Test 1: High CPU on ECS
    print("Test 1: High CPU on ECS Service")
    print("-" * 60)
    insight = generator.analyze_high_cpu(
        resource_type="ECS",
        resource_name="api-service",
        cpu_percent=85,
        additional_context={"task_count": 2, "queue_depth": 1247}
    )
    print(f"Summary: {insight['summary']}")
    print(f"Root Cause: {insight['root_cause']}")
    print(f"Recommendation: {insight['recommendation']['action']}")
    print(f"Cost Impact: {insight['cost_impact']['current']} → {insight['cost_impact']['proposed']}")
    print(f"Savings: {insight['cost_impact']['savings']} ({insight['cost_impact']['savings_percent']}%)")
    print()
    
    # Test 2: Underutilized RDS
    print("Test 2: Underutilized RDS Instance")
    print("-" * 60)
    insight = generator.analyze_underutilized_resource(
        resource_type="RDS",
        resource_name="production-db",
        utilization={"cpu": 23, "memory": 31},
        instance_specs={"instance_class": "db.r5.2xlarge"}
    )
    print(f"Summary: {insight['summary']}")
    print(f"Root Cause: {insight.get('root_cause', 'N/A')}")
    if 'recommendation' in insight:
        print(f"Recommendation: {insight['recommendation']['action']}")
        print(f"Savings: {insight['cost_impact']['savings']}/month")
    print()
    
    # Test 3: Cost Anomaly
    print("Test 3: Cost Anomaly Detection")
    print("-" * 60)
    insight = generator.analyze_cost_anomaly(
        service_name="RDS",
        current_cost=600,
        baseline_cost=400
    )
    print(f"Summary: {insight['summary']}")
    print(f"Increase: {insight['metrics']['increase']} ({insight['metrics']['increase_percent']})")
    print(f"Priority: {insight['priority']}")
    print()
