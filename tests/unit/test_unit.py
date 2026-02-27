"""
Sample Unit Tests for AutoPilot AI

Demonstrates unit testing for individual components.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import MetricData, MetricType, Anomaly, CostImpact
from agents.observability_agent import ObservabilityAgent


class TestObservabilityAgent:
    """Unit tests for Observability Agent"""
    
    def test_detect_anomalies_with_valid_metrics(self):
        """Test anomaly detection with normal and anomalous metrics"""
        agent = ObservabilityAgent()
        
        # Create metrics with one clear anomaly
        metrics = [
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=50.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=52.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=51.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=95.0,  # Anomaly
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
        ]
        
        anomalies = agent.detect_anomalies(metrics, sigma_threshold=2.0)
        
        # Should detect the 95% value as anomaly
        assert len(anomalies) >= 1
        assert any(a.observed_value == 95.0 for a in anomalies)
    
    def test_detect_anomalies_insufficient_data(self):
        """Test anomaly detection with insufficient data points"""
        agent = ObservabilityAgent()
        
        # Only 2 metrics - not enough for statistics
        metrics = [
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=50.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=52.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ),
        ]
        
        anomalies = agent.detect_anomalies(metrics)
        
        # Should not detect anomalies with insufficient data
        assert len(anomalies) == 0
    
    def test_detect_anomalies_no_variance(self):
        """Test anomaly detection when all values are the same"""
        agent = ObservabilityAgent()
        
        # All same values
        metrics = [
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=50.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            ) for _ in range(5)
        ]
        
        anomalies = agent.detect_anomalies(metrics)
        
        # Should not detect anomalies when variance is zero
        assert len(anomalies) == 0


class TestAgentProtocol:
    """Unit tests for agent protocol"""
    
    def test_agent_response_success_requires_insights(self):
        """Test that SUCCESS status requires insights"""
        
        # Should raise error when SUCCESS but no insights
        with pytest.raises(ValueError, match="Successful response must include insights"):
            AgentResponse(
                agent_type=AgentType.OBSERVABILITY,
                task_id="test-123",
                status=TaskStatus.SUCCESS,
                insights=[],  # Empty insights
                data={},
                execution_time_ms=100.0
            )
    
    def test_agent_response_valid_success(self):
        """Test valid SUCCESS response"""
        
        insight = Insight(
            summary="Test insight",
            business_impact="Test impact",
            severity=Severity.LOW,
            recommendations=["Action 1"],
            confidence_score=0.8
        )
        
        response = AgentResponse(
            agent_type=AgentType.OBSERVABILITY,
            task_id="test-123",
            status=TaskStatus.SUCCESS,
            insights=[insight],
            data={},
            execution_time_ms=100.0
        )
        
        assert response.status == TaskStatus.SUCCESS
        assert len(response.insights) == 1
        assert response.insights[0].summary == "Test insight"
    
    def test_insight_requires_recommendations(self):
        """Test that insights must include recommendations"""
        
        with pytest.raises(ValueError, match="must include at least one recommendation"):
            Insight(
                summary="Test",
                business_impact="Impact",
                severity=Severity.LOW,
                recommendations=[],  # Empty recommendations
                confidence_score=0.8
            )


class TestCostImpact:
    """Unit tests for cost impact calculations"""
    
    def test_cost_impact_auto_calculation(self):
        """Test automatic calculation of savings"""
        
        cost_impact = CostImpact(
            current_monthly_cost_inr=50000.0,
            projected_monthly_cost_inr=35000.0,
            savings_monthly_inr=0,  # Will be calculated
            savings_annual_inr=0,    # Will be calculated
            confidence_score=0.85,
            breakdown={}
        )
        
        # Should auto-calculate savings
        assert cost_impact.savings_monthly_inr == 15000.0
        assert cost_impact.savings_annual_inr == 180000.0
    
    def test_cost_impact_negative_costs_rejected(self):
        """Test that negative costs are rejected"""
        
        with pytest.raises(ValueError, match="Cost values cannot be negative"):
            CostImpact(
                current_monthly_cost_inr=-1000.0,
                projected_monthly_cost_inr=500.0,
                savings_monthly_inr=0,
                savings_annual_inr=0,
                confidence_score=0.8
            )
    
    def test_cost_impact_confidence_score_validation(self):
        """Test confidence score validation"""
        
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            CostImpact(
                current_monthly_cost_inr=1000.0,
                projected_monthly_cost_inr=500.0,
                savings_monthly_inr=0,
                savings_annual_inr=0,
                confidence_score=1.5  # Invalid
            )


class TestMetricData:
    """Unit tests for metric data"""
    
    def test_metric_data_validation(self):
        """Test metric data validation"""
        
        metric = MetricData(
            metric_name="CPUUtilization",
            metric_type=MetricType.CPU,
            value=75.5,
            unit="Percent",
            timestamp=datetime.utcnow().isoformat()
        )
        
        assert metric.metric_name == "CPUUtilization"
        assert metric.metric_type == MetricType.CPU
        assert metric.value == 75.5
    
    def test_metric_data_empty_name_rejected(self):
        """Test that empty metric name is rejected"""
        
        with pytest.raises(ValueError, match="Metric name cannot be empty"):
            MetricData(
                metric_name="",
                metric_type=MetricType.CPU,
                value=50.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            )
    
    def test_metric_data_negative_value_rejected(self):
        """Test that negative values are rejected for standard metrics"""
        
        with pytest.raises(ValueError, match="Metric value cannot be negative"):
            MetricData(
                metric_name="CPUUtilization",
                metric_type=MetricType.CPU,
                value=-10.0,
                unit="Percent",
                timestamp=datetime.utcnow().isoformat()
            )


# Run with: pytest tests/unit/test_unit.py -v
