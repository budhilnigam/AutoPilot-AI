import React, { useMemo, useState } from 'react'
import { Calculator, Gauge } from 'lucide-react'

function ScenarioSimulatorPanel() {
  const [currentCost, setCurrentCost] = useState(2500)
  const [reductionPercent, setReductionPercent] = useState(20)
  const [incidentRate, setIncidentRate] = useState(6)

  const projection = useMemo(() => {
    const savings = (currentCost * reductionPercent) / 100
    const projected = currentCost - savings
    const riskScore = Math.max(0, Math.min(100, Math.round(incidentRate * 8 - reductionPercent * 0.5)))
    return { savings, projected, riskScore }
  }, [currentCost, reductionPercent, incidentRate])

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="flex items-center text-sm font-semibold text-slate-800 dark:text-slate-100">
          <Calculator className="mr-2 h-4 w-4 text-blue-600" />
          What-if Scenario Simulator
        </h3>

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          <label className="text-xs text-slate-600 dark:text-slate-300">
            Current Monthly Cost (USD)
            <input
              type="number"
              value={currentCost}
              onChange={(event) => setCurrentCost(Number(event.target.value || 0))}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
            />
          </label>

          <label className="text-xs text-slate-600 dark:text-slate-300">
            Optimization (%)
            <input
              type="range"
              min={0}
              max={60}
              value={reductionPercent}
              onChange={(event) => setReductionPercent(Number(event.target.value))}
              className="mt-2 w-full"
            />
            <span className="text-xs font-semibold text-blue-700 dark:text-blue-300">{reductionPercent}%</span>
          </label>

          <label className="text-xs text-slate-600 dark:text-slate-300">
            Monthly Incident Count
            <input
              type="number"
              value={incidentRate}
              onChange={(event) => setIncidentRate(Number(event.target.value || 0))}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
            />
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-green-200 bg-green-50 p-4 dark:border-green-900 dark:bg-green-950/30">
          <p className="text-xs font-semibold uppercase tracking-wide text-green-700 dark:text-green-300">Projected Spend</p>
          <p className="mt-2 text-2xl font-bold text-green-700 dark:text-green-300">${projection.projected.toFixed(2)}</p>
        </div>

        <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-900 dark:bg-blue-950/30">
          <p className="text-xs font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-300">Estimated Savings</p>
          <p className="mt-2 text-2xl font-bold text-blue-700 dark:text-blue-300">${projection.savings.toFixed(2)}</p>
        </div>

        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950/30">
          <p className="flex items-center text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
            <Gauge className="mr-1 h-3 w-3" />
            Delivery Risk Score
          </p>
          <p className="mt-2 text-2xl font-bold text-amber-700 dark:text-amber-300">{projection.riskScore}/100</p>
        </div>
      </div>
    </div>
  )
}

export default ScenarioSimulatorPanel
