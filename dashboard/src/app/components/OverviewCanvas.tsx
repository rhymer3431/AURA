import { motion } from "motion/react";

import { ExternalServicesPanel } from "./ExternalServicesPanel";
import { NavigationControlPanel } from "./NavigationControlPanel";
import { PipelineFlow } from "./PipelineFlow";
import { RobotViewer } from "./RobotViewer";
import { StatCards } from "./StatCards";
import {
  IpcOrchestrationWidget,
  LogsWidget,
  MemoryWidget,
  PerceptionWidget,
  ProcessesWidget,
  SensorsWidget,
} from "./SystemStatusWidgets";

const sectionTransition = { duration: 0.4, ease: [0.22, 1, 0.36, 1] as const };

export function OverviewCanvas() {
  return (
    <div className="space-y-6">
      <StatCards />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
        <motion.div
          className="flex min-w-0 flex-col gap-6 xl:col-span-8"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={sectionTransition}
        >
          <RobotViewer />
          <PipelineFlow />
          <NavigationControlPanel />
          <ExternalServicesPanel />
          <LogsWidget />
        </motion.div>

        <motion.div
          className="min-w-0 xl:col-span-4"
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ ...sectionTransition, delay: 0.08 }}
        >
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-1">
            <ProcessesWidget />
            <SensorsWidget />
            <PerceptionWidget />
            <MemoryWidget />
            <IpcOrchestrationWidget />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
