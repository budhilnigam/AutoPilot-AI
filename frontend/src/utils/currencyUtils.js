/**
 * Currency formatting utilities for India-First design
 * Formats USD to INR with Indian numbering system (Lakhs/Crores)
 */

// Exchange rate (₹ per $1)
const USD_TO_INR = 83.0

// Indian numbering thresholds
const LAKH = 100000 // 1 Lakh = 1,00,000
const CRORE = 10000000 // 1 Crore = 1,00,00,000

/**
 * Convert USD to INR
 */
export function usdToInr(usdAmount) {
  return usdAmount * USD_TO_INR
}

/**
 * Format number with Indian comma placement
 * e.g., 1,23,45,678 instead of 12,345,678
 */
function indianNumberFormat(num) {
  const str = num.toString()
  
  if (str.length <= 3) {
    return str
  }
  
  // Last 3 digits
  let result = str.slice(-3)
  let remaining = str.slice(0, -3)
  
  // Groups of 2 from right to left
  while (remaining.length > 0) {
    if (remaining.length > 2) {
      result = remaining.slice(-2) + ',' + result
      remaining = remaining.slice(0, -2)
    } else {
      result = remaining + ',' + result
      remaining = ''
    }
  }
  
  return result
}

/**
 * Format amount in INR with Indian numbering system
 * 
 * @param {number} amount - Amount in INR
 * @param {boolean} useIndianUnits - Use Lakh/Crore notation
 * @param {number} decimalPlaces - Number of decimal places
 * @returns {string} Formatted string like "₹12,345" or "₹1.2L"
 */
export function formatInr(amount, useIndianUnits = true, decimalPlaces = 0) {
  if (amount < 0) {
    return `-${formatInr(Math.abs(amount), useIndianUnits, decimalPlaces)}`
  }
  
  if (useIndianUnits) {
    // Use Crores for amounts >= 1 Crore
    if (amount >= CRORE) {
      const value = amount / CRORE
      return `₹${value.toFixed(2)}Cr`
    }
    
    // Use Lakhs for amounts >= 1 Lakh
    if (amount >= LAKH) {
      const value = amount / LAKH
      return `₹${value.toFixed(2)}L`
    }
  }
  
  // Standard formatting for smaller amounts
  const integerPart = Math.floor(amount)
  const formattedInteger = indianNumberFormat(integerPart)
  
  if (decimalPlaces > 0) {
    const decimalPart = (amount - integerPart).toFixed(decimalPlaces).slice(2)
    return `₹${formattedInteger}.${decimalPart}`
  }
  
  return `₹${formattedInteger}`
}

/**
 * Convert USD to INR and format
 *  
 * @param {number} usdAmount - Amount in USD
 * @param {boolean} useIndianUnits - Use Lakh/Crore notation
 * @param {boolean} showUsd - Also show USD amount
 * @returns {string} Formatted INR string, optionally with USD
 */
export function formatUsdAsInr(usdAmount, useIndianUnits = true, showUsd = false) {
  const inrAmount = usdToInr(usdAmount)
  const formatted = formatInr(inrAmount, useIndianUnits)
  
  if (showUsd) {
    return `${formatted} ($${usdAmount.toFixed(2)})`
  }
  
  return formatted
}

/**
 * Format monthly cost in INR
 */
export function formatMonthlyCost(monthlyUsd) {
  const inr = formatUsdAsInr(monthlyUsd)
  return `${inr}/month`
}

/**
 * Format cost savings with percentage
 */
export function formatCostSavings(currentUsd, proposedUsd) {
  const currentInr = usdToInr(currentUsd)
  const proposedInr = usdToInr(proposedUsd)
  const savingsInr = currentInr - proposedInr
  const savingsPercent = currentInr > 0 ? (savingsInr / currentInr * 100) : 0
  
  return {
    current: `${formatInr(currentInr)}/month`,
    proposed: `${formatInr(proposedInr)}/month`,
    savings: `${formatInr(Math.abs(savingsInr))}/month`,
    savingsInr: Math.abs(savingsInr),
    savingsPercent: Math.round(savingsPercent * 10) / 10,
    isSaving: savingsInr > 0
  }
}

/**
 * Format ROI calculation
 */
export function formatRoi(monthlySavingsUsd, implementationCostUsd = 0) {
  const monthlySavingsInr = usdToInr(monthlySavingsUsd)
  const implementationCostInr = usdToInr(implementationCostUsd)
  const annualSavingsInr = monthlySavingsInr * 12
  
  let roiPercent
  let paybackMonths
  
  if (implementationCostInr > 0) {
    roiPercent = ((annualSavingsInr - implementationCostInr) / implementationCostInr) * 100
    paybackMonths = monthlySavingsInr > 0 ? implementationCostInr / monthlySavingsInr : Infinity
  } else {
    roiPercent = Infinity
    paybackMonths = 0
  }
  
  return {
    monthlySavings: `${formatInr(monthlySavingsInr)}/month`,
    monthlySavingsInr,
    implementationCost: formatInr(implementationCostInr),
    implementationCostInr,
    roiPercent: roiPercent === Infinity ? '∞' : Math.round(roiPercent),
    paybackMonths: paybackMonths === Infinity ? 0 : Math.round(paybackMonths * 10) / 10,
    annualSavings: `${formatInr(annualSavingsInr)}/year`,
    annualSavingsInr
  }
}

/**
 * Detect and enhance cost mentions in text
 * Replaces USD amounts with INR equivalents
 * e.g., "$100" → "₹8,300 ($100)"
 */
export function enhanceTextWithInr(text) {
  if (!text) return text
  
  // Match dollar amounts like $100, $1,000, $12.50
  const dollarRegex = /\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)/g
  
  return text.replace(dollarRegex, (match, amount) => {
    const cleanAmount = parseFloat(amount.replace(/,/g, ''))
    const inr = formatUsdAsInr(cleanAmount, true, false)
    return `${inr} (${match})`
  })
}

/**
 * Extract cost impact from insight data and format
 */
export function formatCostImpact(costImpact) {
  if (!costImpact) return null
  
  const formatted = {}
  
  if (costImpact.current) {
    formatted.current = costImpact.current
  }
  
  if (costImpact.proposed) {
    formatted.proposed = costImpact.proposed
  }
  
  if (costImpact.savings) {
    formatted.savings = costImpact.savings
    formatted.savingsPercent = costImpact.savings_percent || costImpact.savingsPercent
  }
  
  if (costImpact.monthly_savings) {
    formatted.monthlySavings = costImpact.monthly_savings
  }
  
  if (costImpact.annual_savings) {
    formatted.annualSavings = costImpact.annual_savings
  }
  
  return formatted
}

/**
 * Highlight INR symbols and amounts in text
 * Returns plain text with markdown emphasis to avoid JSX in .js files
 */
export function highlightInr(text) {
  if (!text) return text
  
  // Match ₹ amounts with various formats
  const inrRegex = /(₹[\d,]+(?:\.\d+)?(?:L|Cr)?(?:\/month|\/year)?)/g

  // Emphasize INR values in markdown-compatible format.
  return text.replace(inrRegex, '**$1**')
}

export default {
  formatInr,
  formatUsdAsInr,
  formatMonthlyCost,
  formatCostSavings,
  formatRoi,
  enhanceTextWithInr,
  formatCostImpact,
  highlightInr,
  usdToInr
}
