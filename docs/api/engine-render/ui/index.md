# engine-render::ui

`engine-render::ui` is ~4541 tokens of signatures — too many to load whole.
Each file below is one part of it; open only the ones you need.

| Module | Digest | ~tok | Public symbols |
|---|---|---|---|
| `engine-render::ui::dock` | [dock.md](dock.md) | 304 | `DockSpace`, `DockStyle`, `DragPanel`, `PanelId`, `Side`, `content()`, `dock()`, `node()`, `panel()`, `select()`, `set_ratio()`, `showing()`, `title()`, `update()` |
| `engine-render::ui::font` | [font.md](font.md) | 157 | `glyph_uv()`, `rasterize_atlas()`, `text_width()` |
| `engine-render::ui::gpu` | [gpu.md](gpu.md) | 311 | `UiGpu`, `advance_staging_slot()`, `draw_secondary()`, `ensure_capacity()`, `last_dirty_words()`, `on_resize()`, `rebind_targets()`, `refresh_textures()`, `scatter_secondary()`, `write_slot()`, `write_staging()` |
| `engine-render::ui::keyboard` | [keyboard.md](keyboard.md) | 110 | `focus()`, `focused()`, `keyboard_captured()`, `set_focus()` |
| `engine-render::ui::list` | [list.md](list.md) | 461 | `DropMark`, `Row`, `RowContent`, `RowList`, `RowStyle`, `bound_row()`, `clicked()`, `content()`, `double_clicked()`, `dragged()`, `dropped()`, `dropped_on()`, `hovered()`, `hovered_at()`, `node()`, `right_clicked()`, `rows()`, `set_drop_mark()`, `set_selected()`, `style()`, `sync()` |
| `engine-render::ui::menu_bar` | [menu_bar.md](menu_bar.md) | 170 | `MenuBar`, `MenuBarStyle`, `node()`, `update()` |
| `engine-render::ui::mod` | [mod.md](mod.md) | 731 | `GroupId`, `PrimId`, `TextId`, `UiCore`, `UiGroup`, `UiQuad`, `UiStyle`, `border()`, `corners()`, `fill()`, `free()`, `global()`, `group()`, `group_count()`, `image()`, `prim_count()`, `radius()`, `rect()`, `set_fill()`, `set_group_clip()`, `set_group_offset()`, `set_group_opacity()`, `set_rect()`, `set_style()`, +10 more |
| `engine-render::ui::popup` | [popup.md](popup.md) | 233 | `MenuStyle`, `PopupStyle`, `close_popup()`, `context_menu()`, `menu_choice()`, `menu_payload()`, `popup()`, `popup_open()` |
| `engine-render::ui::text_field` | [text_field.md](text_field.md) | 267 | `TextFieldStyle`, `changed()`, `cursor()`, `field_text()`, `focus()`, `set_hint()`, `set_text()`, `submitted()`, `text()`, `text_field()` |
| `engine-render::ui::theme` | [theme.md](theme.md) | 158 | `Theme`, `set_theme()`, `theme()` |
| `engine-render::ui::tree` | [tree.md](tree.md) | 837 | `Drag`, `Events`, `NodeId`, `Scrub`, `beyond()`, `claim_drag()`, `click_count()`, `clicked()`, `delta()`, `double_clicked()`, `drag()`, `dragging()`, `dropped()`, `dropped_on()`, `generation()`, `ghost()`, `grab()`, `has()`, `held()`, `hit_test()`, `hovered()`, `image()`, `index()`, `label()`, +22 more |
| `engine-render::ui::tree_view` | [tree_view.md](tree_view.md) | 421 | `DragNode`, `Dropped`, `TreeDrag`, `TreeView`, `clicked()`, `double_clicked()`, `dropped()`, `grab()`, `hovered()`, `invalidate()`, `is_expanded()`, `moved()`, `node()`, `picked_up()`, `reveal()`, `right_clicked()`, `row()`, `set_expanded()`, `sync()`, `visible()`, `with_tree()` |
| `engine-render::ui::viewport` | [viewport.md](viewport.md) | 122 | `Viewport`, `camera()`, `node()`, `update()` |
| `engine-render::ui::widget` | [widget.md](widget.md) | 861 | `Button`, `ButtonStyle`, `Checkbox`, `CheckboxStyle`, `Label`, `Menu`, `Popup`, `RadioGroup`, `RadioStyle`, `Scrollbar`, `ScrollbarStyle`, `Slider`, `SliderStyle`, `StateStyle`, `TabStyle`, `Tabs`, `TextField`, `button()`, `checkbox()`, `checked()`, `fills()`, `node()`, `pane()`, `radio_group()`, +11 more |

[engine index](../../index.md)
