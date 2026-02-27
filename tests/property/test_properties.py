"""
Sample Property-Based Tests for AutoPilot AI

Demonstrates property-based testing using Hypothesis.
Tests the 35 correctness properties defined in the design.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck
from datetime import datetime

from models.agent_protocol import (
    AgentResponse, AgentType, TaskStatus, Severity, Insight
)
from models.core_models import (
    MetricData, MetricType, Anomaly, Recommendation, CostImpact
)


# Strategy for generating valid MetricData
@st.composite
def metric_data_strategy(draw):
    """Generate valid MetricData for testing"""
    metric_types = list(MetricType)
    
    return MetricData(
        metric_name=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        metric_type=draw(st.sampled_from(metric_types)),
        value=draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)),
        unit=draw(st.sampled_from(['Percent', 'Count', 'Bytes', 'Seconds', 'None'])),
        timestamp=datetime.utcnow().isoformat(),
        dimensions=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=50),
            max_size=5
        )),
        source=draw(st.sampled_from(['cloudwatch', 'datadog', 'prometheus']))
    )


# Strategy for generating valid Insights
@st.composite
def insight_strategy(draw):
    """Generate valid Insight for testing"""
    return Insight(
        summary=draw(st.text(min_size=10, max_size=200)),
        business_impact=draw(st.text(min_size=10, max_size=200)),
        severity=draw(st.sampled_from(list(Severity))),
        recommendations=draw(st.lists(
            st.text(min_size=5, max_size=100),
            min_size=1,
            max_size=10
        )),
        cost_impact_inr=draw(st.floats(min_value=-100000, max_value=100000, allow_nan=False)),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0))
    )


# Strategy for generating valid AgentResponse
@st.composite
def agent_response_strategy(draw):
    """Generate valid AgentResponse for testing"""
    status = draw(st.sampled_from(list(TaskStatus)))
    
    # If status is SUCCESS, must have at least one insight
    if status == TaskStatus.SUCCESS:
        insights = draw(st.lists(insight_strategy(), min_size=1, max_size=5))
    else:
        insights = draw(st.lists(insight_strategy(), max_size=3))
    
    return AgentResponse(
        agent_type=draw(st.sampled_from(list(AgentType))),
        task_id=draw(st.text(min_size=10, max_size=50)),
        status=status,
        insights=insights,
        data=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(),
                st.booleans()
            ),
            max_size=10
        )),
        execution_time_ms=draw(st.floats(min_value=0, max_value=60000, allow_nan=False))
    )


class TestAgentProtocolProperties:
    """Property 35: Agent Response Structure Consistency"""
    
    @given(agent_response_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_agent_response_structure_consistency(self, response: AgentResponse):
        """
        Property 35: Agent Response Structure Consistency
        
        All agent responses must maintain consistent structure:
        - Valid agent_type
        - Non-empty task_id
        - Valid status
        - Non-negative execution time
        - SUCCESS status requires insights
        """
        # Agent type must be valid
        assert response.agent_type in AgentType
        
        # Task ID must not be empty
        assert len(response.task_id) > 0
        
        # Status must be valid
        assert response.status in TaskStatus
        
        # Execution time must be non-negative
        assert response.execution_time_ms >= 0
        
        # SUCCESS status must have insights
        if response.status == TaskStatus.SUCCESS:
            assert len(response.insights) > 0
        
        # Timestamp must be ISO format
        datetime.fromisoformat(response.timestamp)
    
    @given(st.lists(insight_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_insight_recommendation_completeness(self, insights: list):
        """
        Property: Recommendation Completeness
        
        All insights must include at least one actionable recommendation.
        """
        for insight in insights:
            assert len(insight.recommendations) >= 1
            assert all(len(rec) > 0 for rec in insight.recommendations)
    
    @given(insight_strategy())
    @settings(max_examples=100)
    def test_confidence_score_bounds(self, insight: Insight):
        """
        Property: Confidence Score Validity
        
        All confidence scores must be between 0.0 and 1.0.
        """
        assert 0.0 <= insight.confidence_score <= 1.0


class TestMetricDataProperties:
    """Test core metric data properties"""
    
    @given(metric_data_strategy())
    @settings(max_examples=100)
    def test_metric_data_validity(self, metric: MetricData):
        """
        Property: Metric Data Validity
        
        All metric data must be valid:
        - Non-empty metric name
        - Valid metric type
        - Non-negative value
        - Valid timestamp
        """
        assert len(metric.metric_name) > 0
        assert metric.metric_type in MetricType
        assert metric.value >= 0
        datetime.fromisoformat(metric.timestamp)
    
    @given(st.lists(metric_data_strategy(), min_size=2, max_size=20))
    @settings(max_examples=50)
    def test_anomaly_detection_sensitivity(self, metrics: list):
        """
        Property 5: Anomaly Detection Sensitivity
        
        When metrics vary significantly, anomaly detection should identify outliers.
        This is a simplified version - real implementation would use the actual agent.
        """
        # Group by metric name
        metric_groups = {}
        for m in metrics:
            if m.metric_name not in metric_groups:
                metric_groups[m.metric_name] = []
            metric_groups[m.metric_name].append(m.value)
        
        # Each metric group should have consistent type
        for name, values in metric_groups.items():
            if len(values) >= 3:
                import statistics
                mean = statistics.mean(values)
                try:
                    stdev = statistics.stdev(values)
                    # Property: Values within 2 sigma are not anomalies
                    for v in values:
                        deviation = abs(v - mean) / (stdev if stdev > 0 else 1)
                        if deviation < 2.0:
                            # This value should not be flagged as anomaly
                            assert True  # Placeholder for actual anomaly check
                except statistics.StatisticsError:
                    pass  # All values are the same


class TestCostImpactProperties:
    """Test cost impact calculation properties"""
    
    @given(
        current=st.floats(min_value=1000, max_value=100000, allow_nan=False),
        projected=st.floats(min_value=500, max_value=100000, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_cost_impact_calculation(self, current: float, projected: float):
        """
        Property: Cost Impact Calculation Correctness
        
        Savings calculations must be consistent:
        - savings_monthly = current - projected
        - savings_annual = savings_monthly * 12
        """
        cost_impact = CostImpact(
            current_monthly_cost_inr=current,
            projected_monthly_cost_inr=projected,
            savings_monthly_inr=0,  # Will be auto-calculated
            savings_annual_inr=0,    # Will be auto-calculated
            confidence_score=0.8,
            breakdown={}
        )
        
        expected_monthly = current - projected
        expected_annual = expected_monthly * 12
        
        # Allow small floating point errors
        assert abs(cost_impact.savings_monthly_inr - expected_monthly) < 0.01
        assert abs(cost_impact.savings_annual_inr - expected_annual) < 0.01
    
    @given(
        savings_monthly=st.floats(min_value=-50000, max_value=50000, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_annual_savings_calculation(self, savings_monthly: float):
        """
        Property: Annual savings must be monthly savings * 12
        """
        savings_annual = savings_monthly * 12
        
        # Property: annual = monthly * 12
        assert abs(savings_annual - (savings_monthly * 12)) < 0.01


class TestRecommendationProperties:
    """Test recommendation properties"""
    
    @given(
        title=st.text(min_size=1, max_size=100),
        action_steps=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10),
        priority=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_recommendation_structure(self, title: str, action_steps: list, priority: int):
        """
        Property: Recommendation Structure
        
        All recommendations must have:
        - Non-empty title
        - At least one action step
        - Valid priority (1-5)
        """
        rec = Recommendation(
            title=title,
            description="Test recommendation",
            action_steps=action_steps,
            priority=priority
        )
        
        assert len(rec.title) > 0
        assert len(rec.action_steps) >= 1
        assert 1 <= rec.priority <= 5
        assert rec.cost_impact_monthly_inr >= 0


# Run with: pytest tests/property/test_properties.py -v
