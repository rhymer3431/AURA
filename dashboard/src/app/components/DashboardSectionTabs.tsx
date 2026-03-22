import { dashboardSectionForPage, type DashboardPageId } from "../navigation";

export function DashboardSectionTabs({
  activePage,
  onNavigate,
}: {
  activePage: DashboardPageId;
  onNavigate: (pageId: DashboardPageId) => void;
}) {
  const section = dashboardSectionForPage(activePage);

  if (section === null || section.items.length < 2) {
    return null;
  }

  return (
    <div className="dashboard-section-tabs dashboard-scroll" role="tablist" aria-label={`${section.title} tabs`}>
      {section.items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="tab"
          aria-selected={activePage === item.id}
          className="dashboard-section-tab"
          data-active={activePage === item.id ? "true" : "false"}
          onClick={() => onNavigate(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
