import React from 'react'
import { AlertTriangle, DollarSign, Rocket, Shield } from 'lucide-react'

function ExecutiveSnapshot({ summary }) {
  const cards = [
    {
      id: 'risk',
      title: 'Operational Risk',
      value: summary.operationalRisk,
      hint: 'Aggregated from alerts and build health',
      icon: AlertTriangle,
      classes: 'border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300',
    },
    {
      id: 'savings',
      title: 'Savings Opportunity',
      value: summary.savingsOpportunity,
      hint: 'Monthly estimate',
      icon: DollarSign,
      classes: 'border-green-200 bg-green-50 text-green-700 dark:border-green-900 dark:bg-green-950/30 dark:text-green-300',
    },
    {
      id: 'delivery',
      title: 'Delivery Health',
      value: summary.deliveryHealth,
      hint: 'Pipeline reliability signal',
      icon: Rocket,
      classes: 'border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950/30 dark:text-blue-300',
    },
    {
      id: 'security',
      title: 'Security Posture',
      value: summary.securityPosture,
      hint: 'Current confidence state',
      icon: Shield,
      classes: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300',
    },
  ]

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Executive Snapshot</h3>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
          A concise strategic view of system stability, cost, and delivery confidence.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {cards.map((card) => {
          const Icon = card.icon
          return (
            <div key={card.id} className={`rounded-xl border p-4 ${card.classes}`}>
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-wide">{card.title}</p>
                <Icon className="h-4 w-4" />
              </div>
              <p className="mt-2 text-2xl font-bold">{card.value}</p>
              <p className="mt-1 text-xs opacity-80">{card.hint}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ExecutiveSnapshot
