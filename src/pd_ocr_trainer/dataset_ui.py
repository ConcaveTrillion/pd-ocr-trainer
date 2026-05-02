"""Dataset kanban UI section for the OCR training interface.

Provides build_dataset_section() which renders the full dataset management
card (pending exports + on-disk data kanban) into the current NiceGUI context.
"""

from collections.abc import Callable

from nicegui import ui

from .dataset_store import ExportManager

COLUMN_DEFS = [
    ("unassigned", "📋 Unassigned", "border-gray-300"),
    ("train", "🔵 Training", "border-blue-400"),
    ("val", "🟢 Validation", "border-teal-400"),
]


def build_dataset_section(
    export_manager: ExportManager,
    model_profile_state: dict,
) -> tuple[Callable[[], None], object]:
    """Build the dataset kanban card into the current NiceGUI column context.

    Returns (refresh_kanban, scope_label_widget).
    The caller is responsible for storing scope_label_widget so that the profile
    selector can update it when the active profile changes.
    """
    with ui.card().classes("w-full"):
        ui.label("📂 Dataset Management").classes("text-lg font-bold")
        scope_label = ui.label("").classes("text-xs text-gray-500")
        ui.label(
            "Auto-populated from the pd-ocr-labeler DocTR export root. "
            "Yellow items are already present in the training/validation datasets. "
            "Drag project rows or individual subfolder chips between columns. "
            "For on-disk pages: click to select, Ctrl/Cmd-click to toggle, Shift-click for range."
        ).classes("text-xs text-gray-500 mb-2")

        copy_detection = {"value": True}
        copy_recognition = {"value": True}

        with ui.row().classes("items-center gap-6 mb-3"):
            ui.checkbox(
                text="Include detection dataset",
                value=True,
                on_change=lambda v: copy_detection.__setitem__("value", bool(v.value)),
            )
            ui.checkbox(
                text="Include recognition dataset",
                value=True,
                on_change=lambda v: copy_recognition.__setitem__("value", bool(v.value)),
            )

        kanban_status = ui.label("").classes("text-xs text-gray-600")
        dragging: dict = {"type": None, "key": None}
        col_containers: dict = {}
        selected_existing_pages: set[tuple[str, str]] = set()
        page_selection_anchor: dict[str, str] = {}

        def _select_existing_page(
            event,
            split_col: str,
            project_id: str,
            page_name: str,
            ordered_pages: list[str],
        ) -> None:
            args = event.args if hasattr(event, "args") and isinstance(event.args, dict) else {}
            shift_pressed = bool(args.get("shiftKey"))
            toggle_pressed = bool(args.get("ctrlKey") or args.get("metaKey"))
            key = (split_col, page_name)
            scope_key = f"{split_col}:{project_id}"

            if shift_pressed and page_selection_anchor.get(scope_key) in ordered_pages:
                start = ordered_pages.index(page_selection_anchor[scope_key])
                end = ordered_pages.index(page_name)
                if start > end:
                    start, end = end, start
                range_keys = {(split_col, p) for p in ordered_pages[start : end + 1]}
                if toggle_pressed:
                    selected_existing_pages.update(range_keys)
                else:
                    selected_existing_pages.difference_update(
                        {selected for selected in selected_existing_pages if selected[0] == split_col}
                    )
                    selected_existing_pages.update(range_keys)
            elif toggle_pressed:
                if key in selected_existing_pages:
                    selected_existing_pages.remove(key)
                else:
                    selected_existing_pages.add(key)
            else:
                selected_existing_pages.clear()
                selected_existing_pages.add(key)

            page_selection_anchor[scope_key] = page_name
            refresh_kanban()

        def handle_drop(target: str) -> None:
            dtype = dragging.get("type")
            key = dragging.get("key")
            if dtype == "export" and isinstance(key, str):
                export_manager.assign(key, target)
            elif dtype == "export_page" and isinstance(key, tuple) and len(key) == 2:
                export_key, page_name = key
                if isinstance(export_key, str) and isinstance(page_name, str):
                    export_manager.assign_page(export_key, page_name, target)
            elif dtype == "project_pages" and isinstance(key, list):
                page_tuples = [k for k in key if isinstance(k, tuple) and len(k) == 2]
                normalized = [(k, p) for k, p in page_tuples if isinstance(k, str) and isinstance(p, str)]
                if normalized:
                    export_manager.assign_pages(normalized, target)
            elif dtype == "project" and isinstance(key, str):
                export_manager.assign_project(key, target)
            elif dtype == "existing" and isinstance(key, str):
                from_split = dragging.get("from_split")
                if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                    export_manager.move_existing_project(key, from_split, target)
            elif dtype == "existing_page" and isinstance(key, str):
                from_split = dragging.get("from_split")
                if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                    export_manager.move_existing_page(key, from_split, target)
            elif dtype == "existing_page" and isinstance(key, list):
                from_split = dragging.get("from_split")
                if isinstance(from_split, str) and from_split != target and target in ("train", "val"):
                    for page_name in key:
                        if isinstance(page_name, str):
                            export_manager.move_existing_page(page_name, from_split, target)
                    selected_existing_pages.difference_update(
                        {selected for selected in selected_existing_pages if selected[0] == from_split}
                    )
            dragging["type"] = None
            dragging["key"] = None
            dragging["from_split"] = None
            refresh_kanban()

        def render_column(col_id: str, container) -> None:
            container.clear()
            by_split_pages = export_manager.get_export_pages_by_split()
            pending_projects = dict(sorted(by_split_pages.get(col_id, {}).items()))

            existing_projects: dict[str, int] = {}
            if col_id == "train":
                existing_projects = export_manager.get_existing_projects(export_manager.split_root("train"))
            elif col_id == "val":
                existing_projects = export_manager.get_existing_projects(export_manager.split_root("val"))

            has_content = pending_projects or existing_projects
            with container:
                if not has_content:
                    ui.label("(empty — drop items here)").classes("text-xs text-gray-400 italic p-2")
                    return

                # --- Pending (from labeler export root) ---
                for project_id, pages_by_key in pending_projects.items():
                    page_count = sum(len(pages) for pages in pages_by_key.values())
                    exp_label = f"{project_id}  ·  {page_count} pages  [export]"
                    with ui.expansion(exp_label).classes(
                        "w-full mb-1 bg-slate-50 border border-slate-200 rounded"
                    ) as exp_card:
                        exp_card.props("dense")
                        with ui.column().classes("w-full gap-1 pl-2"):
                            with ui.card().classes(
                                "w-full mb-1 px-2 py-1 border border-dashed rounded shadow-none bg-slate-100 text-slate-700 cursor-grab"
                            ) as project_drag_row:
                                project_drag_row.props("draggable=true")
                                project_drag_row.on(
                                    "dragstart",
                                    lambda e, pages=pages_by_key: (
                                        dragging.__setitem__("type", "project_pages"),
                                        dragging.__setitem__(
                                            "key",
                                            [
                                                (k, page_name)
                                                for k, page_names in pages.items()
                                                for page_name in page_names
                                            ],
                                        ),
                                    ),
                                )
                                ui.label("Drag to move all pages").classes("text-xs")
                            for key, pages in pages_by_key.items():
                                changed = export_manager.is_changed(key)
                                for page_name in pages:
                                    row_cls = "w-full mb-1 px-2 py-1 border rounded shadow-none cursor-grab " + (
                                        "bg-yellow-100 border-yellow-400 text-yellow-800"
                                        if changed
                                        else "bg-white border-slate-200 text-slate-600"
                                    )
                                    with ui.card().classes(row_cls) as export_row:
                                        export_row.props("draggable=true")
                                        export_row.on(
                                            "dragstart.stop",
                                            lambda e, k=key, p=page_name: (
                                                dragging.__setitem__("type", "export_page"),
                                                dragging.__setitem__("key", (k, p)),
                                            ),
                                        )
                                        ui.label(page_name).classes("text-xs font-mono")

                # --- Already present in ml-training / ml-validation ---
                if existing_projects:
                    if col_id in ("train", "val") and pending_projects:
                        ui.separator()
                    for project_id, page_count in existing_projects.items():
                        pages = export_manager.get_existing_pages(
                            export_manager.split_root("train" if col_id == "train" else "val"),
                            project_id,
                        )
                        exp_label = f"{project_id}  ·  {page_count} pages  [on disk]"
                        with ui.expansion(exp_label).classes(
                            "w-full mb-1 bg-slate-50 border border-slate-200 rounded"
                        ) as exp_card:
                            exp_card.props("dense")
                            with ui.column().classes("w-full gap-0 pl-2"):
                                with ui.card().classes(
                                    "w-full mb-1 px-2 py-1 border border-dashed rounded shadow-none bg-slate-100 text-slate-700 cursor-grab"
                                ) as project_drag_row:
                                    project_drag_row.props("draggable=true")
                                    project_drag_row.on(
                                        "dragstart",
                                        lambda e, p=project_id, s=col_id: (
                                            dragging.__setitem__("type", "existing"),
                                            dragging.__setitem__("key", p),
                                            dragging.__setitem__("from_split", s),
                                        ),
                                    )
                                    ui.label("Drag to move all pages").classes("text-xs")
                                for img_name in pages:
                                    selected = (col_id, img_name) in selected_existing_pages
                                    selected_cls = (
                                        "bg-blue-100 border-blue-400 text-blue-800"
                                        if selected
                                        else "bg-white border-slate-200 text-slate-600"
                                    )
                                    with ui.card().classes(
                                        "w-full mb-1 px-2 py-1 border rounded shadow-none cursor-grab " + selected_cls
                                    ) as page_row:
                                        page_row.props("draggable=true")
                                        page_row.on(
                                            "click",
                                            lambda e, p=img_name, s=col_id, pr=project_id, ordered=pages: (
                                                _select_existing_page(e, s, pr, p, ordered)
                                            ),
                                        )
                                        page_row.on(
                                            "dragstart.stop",
                                            lambda e, p=img_name, s=col_id: (
                                                dragging.__setitem__("type", "existing_page"),
                                                dragging.__setitem__(
                                                    "key",
                                                    sorted(
                                                        [name for split, name in selected_existing_pages if split == s]
                                                    )
                                                    if (s, p) in selected_existing_pages
                                                    and any(split == s for split, _ in selected_existing_pages)
                                                    else p,
                                                ),
                                                dragging.__setitem__("from_split", s),
                                            ),
                                        )
                                        ui.label(img_name).classes("text-xs font-mono")

        def refresh_kanban() -> None:
            for col_id, container in col_containers.items():
                render_column(col_id, container)

        with ui.row().classes("w-full gap-4"):
            for col_id, col_title, col_border_cls in COLUMN_DEFS:
                with ui.card().classes(f"flex-1 min-h-40 border-2 {col_border_cls}") as column_card:
                    with ui.row().classes("items-center justify-between w-full mb-1"):
                        ui.label(col_title).classes("font-semibold text-sm")
                        if col_id != "unassigned":

                            def _make_clear(t: str):
                                def _clear():
                                    export_manager.clear_split(t)
                                    refresh_kanban()

                                return _clear

                            ui.button("Clear", on_click=_make_clear(col_id)).props("size=xs color=negative flat")
                    column_card.on("dragover.prevent", lambda e: None)
                    column_card.on("drop", lambda e, t=col_id: handle_drop(t))
                    drop_area = ui.column().classes("w-full gap-1 min-h-16")
                    col_containers[col_id] = drop_area

        with ui.row().classes("items-center gap-4 mt-3"):
            ui.button(
                "🔄 Refresh",
                on_click=lambda: (
                    export_manager.scan(),
                    refresh_kanban(),
                    kanban_status.set_text("Refreshed exports from labeler."),
                ),
            ).props("color=secondary")

            def save_assignments() -> None:
                try:
                    result = export_manager.save_assignments(
                        include_detection=copy_detection["value"],
                        include_recognition=copy_recognition["value"],
                    )
                    count = result.get("copied", 0)
                    if count == 0:
                        ui.notify("No assigned exports to copy.", type="warning")
                        return
                    refresh_kanban()
                    kanban_status.set_text(
                        f"💾 Copied {count} export task(s) into '{export_manager.active_profile}' datasets."
                    )
                    ui.notify("Assignments saved.", type="positive")
                except Exception as e:
                    ui.notify(str(e), type="negative")

            ui.button("💾 Copy to Datasets", on_click=save_assignments).props("color=primary")

        refresh_kanban()

    return refresh_kanban, scope_label
