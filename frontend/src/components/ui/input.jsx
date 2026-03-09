import React from 'react'
import { cn } from '../../lib/utils'

export const Input = React.forwardRef(function Input({ className, ...props }, ref) {
  return (
    <input
      ref={ref}
      className={cn(
        'h-10 w-full rounded-xl border border-slate-300 bg-white px-3 text-sm text-slate-900 outline-none transition-all placeholder:text-slate-500 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-200 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-400 dark:focus:ring-cyan-900',
        className,
      )}
      {...props}
    />
  )
})
