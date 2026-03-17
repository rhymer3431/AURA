import { NavigationControlPanel } from "./NavigationControlPanel";
import { OccupancyMapPanel } from "./OccupancyMapPanel";

export function NavigationPage() {
  return (
    <div className="space-y-6">
      <NavigationControlPanel />
      <OccupancyMapPanel showRouteDebug={false} />
    </div>
  );
}
