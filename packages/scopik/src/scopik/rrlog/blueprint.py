"""Dashboard layout: tabbed, review-first.

Design intent: opening the viewer should answer "how wrong is the model and
where" without hunting. Tabs keep each screen to one question:

- Overview: every compare pair as a real-vs-sim overlay, summary + events.
- One tab per pair: the overlay large, residual + rolling RMSE beneath.
- Telemetry: raw signal groups, for context digging.
"""

from __future__ import annotations

import rerun.blueprint as rrb


def _overlay_view(name: str) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        origin=f"/compare/{name}",
        contents=[f"+ /compare/{name}/real", f"+ /compare/{name}/sim"],
        name=name,
    )


def _error_view(name: str) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
        origin=f"/compare/{name}",
        contents=[f"+ /compare/{name}/residual", f"+ /compare/{name}/rolling_rmse"],
        name=f"{name} error",
    )


def gap_blueprint(
    pair_names: list[str],
    signal_groups: list[str],
    has_events: bool,
    has_diagnosis: bool = False,
) -> rrb.Blueprint:
    side_panels: list[rrb.ContainerLike] = []
    if has_diagnosis:
        side_panels.append(rrb.TextDocumentView(origin="/diagnosis", name="Diagnosis"))
    side_panels.append(rrb.TextDocumentView(origin="/summary", name="Gap summary"))
    if has_events:
        side_panels.append(rrb.TextLogView(origin="/events", name="Events"))
    side = rrb.Vertical(*side_panels, name="Summary")

    telemetry_grid = rrb.Grid(
        *[
            rrb.TimeSeriesView(origin=f"/real/signals/{group}", name=group)
            for group in signal_groups
        ],
        name="Telemetry",
    )

    tabs: list[rrb.ContainerLike] = []
    if pair_names:
        overview = rrb.Horizontal(
            rrb.Grid(*[_overlay_view(name) for name in pair_names]),
            side,
            column_shares=[3, 1],
            name="Overview",
        )
        tabs.append(overview)
        for name in pair_names:
            tabs.append(
                rrb.Vertical(
                    _overlay_view(name),
                    _error_view(name),
                    row_shares=[3, 2],
                    name=name,
                )
            )
        tabs.append(telemetry_grid)
    else:
        tabs.append(rrb.Horizontal(telemetry_grid, side, column_shares=[3, 1], name="Telemetry"))

    return rrb.Blueprint(
        rrb.Tabs(*tabs, active_tab=0),
        rrb.TimePanel(state="expanded"),
        collapse_panels=False,
    )


def send(blueprint: rrb.Blueprint) -> None:
    import rerun as rr

    rr.send_blueprint(blueprint)
