import type { LucideIcon } from "lucide-react";

import { cn } from "./ui/utils";

export type ConsoleTone = "cyan" | "emerald" | "amber" | "coral" | "violet" | "slate";

const badgeToneClasses: Record<ConsoleTone, string> = {
  cyan: "border-[color:var(--tone-cyan-border)] bg-[var(--tone-cyan-bg)] text-[var(--tone-cyan-fg)]",
  emerald: "border-[color:var(--tone-emerald-border)] bg-[var(--tone-emerald-bg)] text-[var(--tone-emerald-fg)]",
  amber: "border-[color:var(--tone-amber-border)] bg-[var(--tone-amber-bg)] text-[var(--tone-amber-fg)]",
  coral: "border-[color:var(--tone-coral-border)] bg-[var(--tone-coral-bg)] text-[var(--tone-coral-fg)]",
  violet: "border-[color:var(--tone-violet-border)] bg-[var(--tone-violet-bg)] text-[var(--tone-violet-fg)]",
  slate: "border-[color:var(--tone-slate-border)] bg-[var(--tone-slate-bg)] text-[var(--tone-slate-fg)]",
};

const dotToneClasses: Record<ConsoleTone, string> = {
  cyan: "bg-[var(--signal-cyan)]",
  emerald: "bg-[var(--signal-emerald)]",
  amber: "bg-[var(--signal-amber)]",
  coral: "bg-[var(--signal-coral)]",
  violet: "bg-[var(--signal-violet)]",
  slate: "bg-[var(--signal-slate)]",
};

const railToneVars: Record<ConsoleTone, string> = {
  cyan: "var(--signal-cyan)",
  emerald: "var(--signal-emerald)",
  amber: "var(--signal-amber)",
  coral: "var(--signal-coral)",
  violet: "var(--signal-violet)",
  slate: "var(--signal-slate)",
};

export function ConsolePanel({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("dashboard-panel", className)}>{children}</div>;
}

export function ConsoleInset({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("dashboard-inset", className)}>{children}</div>;
}

export function ConsoleBadge({
  tone = "slate",
  children,
  className,
  dot = true,
}: {
  tone?: ConsoleTone;
  children: React.ReactNode;
  className?: string;
  dot?: boolean;
}) {
  return (
    <span className={cn("dashboard-pill", badgeToneClasses[tone], className)}>
      {dot ? <span className={cn("size-1.5 rounded-full", dotToneClasses[tone])} /> : null}
      {children}
    </span>
  );
}

export function ConsoleSectionTitle({
  icon: Icon,
  eyebrow,
  title,
  description,
  action,
  className,
}: {
  icon: LucideIcon;
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-3", className)}>
      <div className="flex items-start gap-3">
        <div className="dashboard-icon-shell mt-0.5">
          <Icon className="size-4" />
        </div>
        <div>
          {eyebrow ? <div className="dashboard-eyebrow mb-1">{eyebrow}</div> : null}
          <h3 className="dashboard-title">{title}</h3>
          {description ? <p className="dashboard-subtitle mt-1">{description}</p> : null}
        </div>
      </div>
      {action}
    </div>
  );
}

export function ConsoleInfoRow({
  label,
  value,
  className,
  valueClassName,
  mono = false,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  className?: string;
  valueClassName?: string;
  mono?: boolean;
}) {
  return (
    <div className={cn("flex items-center justify-between gap-3 text-[11px]", className)}>
      <span className="dashboard-meta">{label}</span>
      <span className={cn("dashboard-data-value", mono && "dashboard-mono", valueClassName)}>{value}</span>
    </div>
  );
}

export function ConsoleMetricCard({
  label,
  value,
  meta,
  tone = "slate",
  className,
  valueClassName,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  meta?: React.ReactNode;
  tone?: ConsoleTone;
  className?: string;
  valueClassName?: string;
}) {
  return (
    <div
      className={cn("dashboard-kpi", className)}
      style={{ "--kpi-tone": railToneVars[tone] } as React.CSSProperties}
    >
      <div className="dashboard-label">{label}</div>
      <div className={cn("dashboard-value mt-4", valueClassName)}>{value}</div>
      {meta ? <div className="dashboard-micro mt-3">{meta}</div> : null}
    </div>
  );
}

export function toneFromStatusTone(tone: string): ConsoleTone {
  if (tone === "green") {
    return "emerald";
  }
  if (tone === "amber") {
    return "amber";
  }
  if (tone === "red") {
    return "coral";
  }
  if (tone === "blue") {
    return "cyan";
  }
  return "slate";
}
