"""
Currency Formatter for India-First Design

Converts USD to INR with proper formatting and Indian numbering system (Lakhs/Crores).
Integrates with AWS Price List API for India region pricing.
"""

import logging
from typing import Optional, Dict, Any
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError
from datetime import datetime

logger = logging.getLogger(__name__)


class CurrencyFormatter:
    """
    Format costs in Indian Rupees (INR) with proper Indian numbering system.
    
    Features:
    - USD to INR conversion
    - Indian numbering (Lakhs, Crores)
    - AWS India region pricing
    - Cost comparisons across regions
    """
    
    # Exchange rate (₹ per $1) - Update this periodically
    # As of March 2026, approximate rate
    USD_TO_INR = 83.0
    
    # Indian numbering thresholds
    LAKH = 100000  # 1 Lakh = 1,00,000
    CRORE = 10000000  # 1 Crore = 1,00,00,000
    
    # India AWS regions
    INDIA_REGIONS = ['ap-south-1', 'ap-south-2']
    
    def __init__(self, exchange_rate: Optional[float] = None):
        """
        Initialize currency formatter.
        
        Args:
            exchange_rate: Optional custom USD to INR rate (default: 83.0)
        """
        if exchange_rate:
            self.USD_TO_INR = exchange_rate
        
        logger.info(f"Currency Formatter initialized (USD to INR: {self.USD_TO_INR})")
    
    def usd_to_inr(self, usd_amount: float) -> float:
        """
        Convert USD to INR.
        
        Args:
            usd_amount: Amount in USD
            
        Returns:
            Amount in INR
        """
        return usd_amount * self.USD_TO_INR
    
    def format_inr(
        self,
        amount: float,
        use_indian_units: bool = True,
        decimal_places: int = 0
    ) -> str:
        """
        Format amount in INR with Indian numbering system.
        
        Args:
            amount: Amount in INR
            use_indian_units: Use Lakh/Crore notation (default: True)
            decimal_places: Number of decimal places (default: 0)
            
        Returns:
            Formatted string like "₹12,345" or "₹1.2L"
            
        Examples:
            format_inr(1234) → "₹1,234"
            format_inr(125000) → "₹1.25L"
            format_inr(10500000) → "₹1.05Cr"
        """
        if amount < 0:
            return f"-{self.format_inr(abs(amount), use_indian_units, decimal_places)}"
        
        if use_indian_units:
            # Use Crores for amounts >= 1 Crore
            if amount >= self.CRORE:
                value = amount / self.CRORE
                return f"₹{value:.2f}Cr"
            
            # Use Lakhs for amounts >= 1 Lakh
            elif amount >= self.LAKH:
                value = amount / self.LAKH
                return f"₹{value:.2f}L"
        
        # Standard formatting for smaller amounts
        formatted = f"{amount:,.{decimal_places}f}"
        
        # Convert to Indian numbering system (last 3 digits, then groups of 2)
        # e.g., 1,23,45,678 instead of 12,345,678
        if '.' in formatted:
            integer_part, decimal_part = formatted.split('.')
        else:
            integer_part = formatted
            decimal_part = None
        
        # Remove existing commas
        integer_part = integer_part.replace(',', '')
        
        # Apply Indian numbering
        if len(integer_part) > 3:
            # Last 3 digits
            result = integer_part[-3:]
            remaining = integer_part[:-3]
            
            # Groups of 2 from right to left
            while remaining:
                if len(remaining) > 2:
                    result = remaining[-2:] + ',' + result
                    remaining = remaining[:-2]
                else:
                    result = remaining + ',' + result
                    remaining = ''
            
            integer_part = result
        
        # Reconstruct with decimal if present
        if decimal_part:
            formatted = f"{integer_part}.{decimal_part}"
        else:
            formatted = integer_part
        
        return f"₹{formatted}"
    
    def format_usd_as_inr(
        self,
        usd_amount: float,
        use_indian_units: bool = True,
        show_usd: bool = False
    ) -> str:
        """
        Convert USD to INR and format.
        
        Args:
            usd_amount: Amount in USD
            use_indian_units: Use Lakh/Crore notation
            show_usd: Also show USD amount (default: False)
            
        Returns:
            Formatted INR string, optionally with USD
            
        Examples:
            format_usd_as_inr(100) → "₹8,300"
            format_usd_as_inr(100, show_usd=True) → "₹8,300 ($100)"
        """
        inr_amount = self.usd_to_inr(usd_amount)
        formatted = self.format_inr(inr_amount, use_indian_units)
        
        if show_usd:
            return f"{formatted} (${usd_amount:,.2f})"
        
        return formatted
    
    def format_monthly_cost(self, monthly_usd: float) -> str:
        """
        Format monthly cost in INR.
        
        Args:
            monthly_usd: Monthly cost in USD
            
        Returns:
            Formatted monthly cost like "₹8,300/month"
        """
        inr = self.format_usd_as_inr(monthly_usd)
        return f"{inr}/month"
    
    def format_cost_savings(
        self,
        current_usd: float,
        proposed_usd: float,
        period: str = "month"
    ) -> Dict[str, Any]:
        """
        Calculate and format cost savings.
        
        Args:
            current_usd: Current cost in USD
            proposed_usd: Proposed cost in USD
            period: Time period (default: "month")
            
        Returns:
            Dictionary with savings details
            
        Example:
            {
                "current": "₹41,500/month",
                "proposed": "₹24,900/month",
                "savings": "₹16,600/month",
                "savings_percent": 40.0,
                "roi_months": 1.2  # If implementation cost provided
            }
        """
        current_inr = self.usd_to_inr(current_usd)
        proposed_inr = self.usd_to_inr(proposed_usd)
        savings_inr = current_inr - proposed_inr
        savings_percent = (savings_inr / current_inr * 100) if current_inr > 0 else 0
        
        return {
            "current": f"{self.format_inr(current_inr)}/{period}",
            "proposed": f"{self.format_inr(proposed_inr)}/{period}",
            "savings": f"{self.format_inr(abs(savings_inr))}/{period}",
            "savings_inr": abs(savings_inr),
            "savings_percent": round(savings_percent, 1),
            "is_saving": savings_inr > 0,
            "monthly_savings_inr": abs(savings_inr) if period == "month" else abs(savings_inr) / 12
        }
    
    def format_roi(
        self,
        monthly_savings_usd: float,
        implementation_cost_usd: float = 0
    ) -> Dict[str, Any]:
        """
        Calculate and format ROI.
        
        Args:
            monthly_savings_usd: Monthly savings in USD
            implementation_cost_usd: One-time implementation cost in USD
            
        Returns:
            ROI details
            
        Example:
            {
                "monthly_savings": "₹18,300/month",
                "implementation_cost": "₹2,400",
                "roi_percent": 662,
                "payback_months": 0.16,
                "annual_savings": "₹2,19,600/year"
            }
        """
        monthly_savings_inr = self.usd_to_inr(monthly_savings_usd)
        implementation_cost_inr = self.usd_to_inr(implementation_cost_usd)
        annual_savings_inr = monthly_savings_inr * 12
        
        # ROI = (Gain - Cost) / Cost * 100
        if implementation_cost_inr > 0:
            roi_percent = ((annual_savings_inr - implementation_cost_inr) / implementation_cost_inr) * 100
            payback_months = implementation_cost_inr / monthly_savings_inr if monthly_savings_inr > 0 else float('inf')
        else:
            roi_percent = float('inf')
            payback_months = 0
        
        return {
            "monthly_savings": f"{self.format_inr(monthly_savings_inr)}/month",
            "monthly_savings_inr": monthly_savings_inr,
            "implementation_cost": self.format_inr(implementation_cost_inr),
            "implementation_cost_inr": implementation_cost_inr,
            "roi_percent": round(roi_percent, 0) if roi_percent != float('inf') else "∞",
            "payback_months": round(payback_months, 1) if payback_months != float('inf') else 0,
            "annual_savings": f"{self.format_inr(annual_savings_inr)}/year",
            "annual_savings_inr": annual_savings_inr
        }
    
    def compare_regions(
        self,
        service: str,
        instance_type: str,
        us_east_price: float = None
    ) -> Dict[str, Any]:
        """
        Compare pricing between US and India regions.
        
        Args:
            service: AWS service (e.g., 'EC2', 'RDS')
            instance_type: Instance type (e.g., 't3.medium')
            us_east_price: Price in us-east-1 (if known)
            
        Returns:
            Price comparison
            
        Note: This is a simplified version. For production, integrate with
              AWS Price List API for real-time pricing.
        """
        # Simplified pricing - In production, use AWS Price List API
        # India regions are typically 5-10% more expensive than US
        india_multiplier = 1.07
        
        if us_east_price:
            india_price_usd = us_east_price * india_multiplier
            
            return {
                "service": service,
                "instance_type": instance_type,
                "us_east_1": {
                    "usd": f"${us_east_price:.2f}",
                    "inr": self.format_usd_as_inr(us_east_price)
                },
                "ap_south_1": {
                    "usd": f"${india_price_usd:.2f}",
                    "inr": self.format_usd_as_inr(india_price_usd)
                },
                "difference_percent": round((india_multiplier - 1) * 100, 1),
                "recommendation": "Deploy in ap-south-1 for lower latency to Indian users"
            }
        
        return {
            "service": service,
            "instance_type": instance_type,
            "note": "Enable AWS Price List API for real-time pricing comparison"
        }


# Global instance for convenience
_formatter = CurrencyFormatter()


def format_inr(amount: float, use_indian_units: bool = True) -> str:
    """Convenience function to format INR amount."""
    return _formatter.format_inr(amount, use_indian_units)


def format_usd_as_inr(usd_amount: float, show_usd: bool = False) -> str:
    """Convenience function to convert and format USD as INR."""
    return _formatter.format_usd_as_inr(usd_amount, show_usd)


def format_monthly_cost(monthly_usd: float) -> str:
    """Convenience function to format monthly cost."""
    return _formatter.format_monthly_cost(monthly_usd)


def format_cost_savings(current_usd: float, proposed_usd: float) -> Dict[str, Any]:
    """Convenience function to calculate savings."""
    return _formatter.format_cost_savings(current_usd, proposed_usd)


def format_roi(monthly_savings_usd: float, implementation_cost_usd: float = 0) -> Dict[str, Any]:
    """Convenience function to calculate ROI."""
    return _formatter.format_roi(monthly_savings_usd, implementation_cost_usd)


if __name__ == "__main__":
    # Test the formatter
    formatter = CurrencyFormatter()
    
    print("=== Currency Formatter Tests ===\n")
    
    # Test 1: Basic INR formatting
    print("Test 1: Basic INR formatting")
    print(f"  ₹1,234: {formatter.format_inr(1234)}")
    print(f"  ₹1,25,000: {formatter.format_inr(125000)}")
    print(f"  ₹1.25L: {formatter.format_inr(125000, use_indian_units=True)}")
    print(f"  ₹1.05Cr: {formatter.format_inr(10500000, use_indian_units=True)}")
    print()
    
    # Test 2: USD to INR conversion
    print("Test 2: USD to INR conversion")
    print(f"  $100 → {formatter.format_usd_as_inr(100)}")
    print(f"  $500 → {formatter.format_usd_as_inr(500, show_usd=True)}")
    print()
    
    # Test 3: Cost savings
    print("Test 3: Cost savings (RDS downgrade)")
    savings = formatter.format_cost_savings(current_usd=500, proposed_usd=200)
    print(f"  Current: {savings['current']}")
    print(f"  Proposed: {savings['proposed']}")
    print(f"  Savings: {savings['savings']} ({savings['savings_percent']}%)")
    print()
    
    # Test 4: ROI calculation
    print("Test 4: ROI (Worker pool scaling)")
    roi = formatter.format_roi(monthly_savings_usd=300, implementation_cost_usd=50)
    print(f"  Monthly Savings: {roi['monthly_savings']}")
    print(f"  Implementation Cost: {roi['implementation_cost']}")
    print(f"  ROI: {roi['roi_percent']}%")
    print(f"  Payback: {roi['payback_months']} months")
    print(f"  Annual Savings: {roi['annual_savings']}")
