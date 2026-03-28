import { DEFAULT_DASHBOARD_PAGE, parseDashboardPageId } from "./navigation";

describe("navigation route parsing", () => {
  it("keeps the new workspace ids intact", () => {
    expect(parseDashboardPageId("#/live-loop")).toBe("live-loop");
    expect(parseDashboardPageId("#/logs-replay")).toBe("logs-replay");
  });

  it("redirects legacy hashes to their new workspace ids", () => {
    expect(parseDashboardPageId("#/planner-control")).toBe("live-loop");
    expect(parseDashboardPageId("#/artifacts-storage")).toBe("runtime-health-recovery");
    expect(parseDashboardPageId("#/occupancy-map")).toBe("spatial-memory-map");
  });

  it("falls back to the default page for unknown hashes", () => {
    expect(parseDashboardPageId("#/unknown-page")).toBe(DEFAULT_DASHBOARD_PAGE);
  });
});
