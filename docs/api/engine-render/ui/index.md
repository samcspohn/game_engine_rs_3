# engine-render::ui

`engine-render::ui` is ~6138 tokens of signatures — too many to load whole.
Each file below is one part of it; open only the ones you need.

| Module | Digest | ~tok | Public symbols |
|---|---|---|---|
| `engine-render::ui::dock` | [dock.md](dock.md) | 425 | `DockSpace`, `DockStyle`, `DragPanel`, `Layout`, `PanelId`, `Side`, `apply()`, `changed()`, `close()`, `content()`, `dock()`, `layout()`, `listen()`, `node()`, `on_change()`, `on_close()`, `panel()`, `select()`, `set_closable()`, `set_ratio()`, `showing()`, `title()` |
| `engine-render::ui::events` | [events.md](events.md) | 563 | `Event`, `Keep`, `TickId`, `UiRef`, `Value`, `dispatched()`, `every()`, `node()`, `notify()`, `on()`, `on_change()`, `on_click()`, `on_double_click()`, `on_drop()`, `on_hover()`, `on_notify()`, `on_resize()`, `on_right_click()`, `on_submit()`, `own()`, `stop()`, `with()` |
| `engine-render::ui::font` | [font.md](font.md) | 157 | `glyph_uv()`, `rasterize_atlas()`, `text_width()` |
| `engine-render::ui::gpu` | [gpu.md](gpu.md) | 311 | `UiGpu`, `advance_staging_slot()`, `draw_secondary()`, `ensure_capacity()`, `last_dirty_words()`, `on_resize()`, `rebind_targets()`, `refresh_textures()`, `scatter_secondary()`, `write_slot()`, `write_staging()` |
| `engine-render::ui::keyboard` | [keyboard.md](keyboard.md) | 110 | `focus()`, `focused()`, `keyboard_captured()`, `set_focus()` |
| `engine-render::ui::list` | [list.md](list.md) | 461 | `DropMark`, `Row`, `RowContent`, `RowList`, `RowStyle`, `bound_row()`, `clicked()`, `content()`, `double_clicked()`, `dragged()`, `dropped()`, `dropped_on()`, `hovered()`, `hovered_at()`, `node()`, `right_clicked()`, `rows()`, `set_drop_mark()`, `set_selected()`, `style()`, `sync()` |
| `engine-render::ui::menu_bar` | [menu_bar.md](menu_bar.md) | 200 | `BarMenu`, `MenuBar`, `MenuBarStyle`, `item()`, `menu()`, `node()` |
| `engine-render::ui::mod` | [mod.md](mod.md) | 816 | `GroupId`, `Paint`, `PrimId`, `TextId`, `UiCore`, `UiGroup`, `UiQuad`, `UiStyle`, `border()`, `corners()`, `defer()`, `fill()`, `free()`, `global()`, `group()`, `group_count()`, `image()`, `keyboard_captured()`, `paint()`, `pointer_captured()`, `prim_count()`, `radius()`, `rect()`, `set_fill()`, +15 more |
| `engine-render::ui::popup` | [popup.md](popup.md) | 247 | `MenuStyle`, `PopupStyle`, `close_popup()`, `context_menu()`, `item()`, `on_close()`, `popup()`, `popup_open()` |
| `engine-render::ui::text_field` | [text_field.md](text_field.md) | 267 | `TextFieldStyle`, `changed()`, `cursor()`, `field_text()`, `focus()`, `set_hint()`, `set_text()`, `submitted()`, `text()`, `text_field()` |
| `engine-render::ui::theme` | [theme.md](theme.md) | 158 | `Theme`, `set_theme()`, `theme()` |
| `engine-render::ui::tree` | [tree.md](tree.md) | 957 | `Drag`, `Events`, `NodeId`, `Scrub`, `beyond()`, `claim_drag()`, `click_count()`, `clicked()`, `delta()`, `double_clicked()`, `drag()`, `dragging()`, `dropped()`, `dropped_on()`, `frame()`, `generation()`, `ghost()`, `grab()`, `has()`, `held()`, `hit_test()`, `hovered()`, `image()`, `index()`, +24 more |
| `engine-render::ui::tree_view` | [tree_view.md](tree_view.md) | 463 | `DragNode`, `Dropped`, `TreeDrag`, `TreeView`, `clicked()`, `double_clicked()`, `dropped()`, `dropped_on()`, `grab()`, `hovered()`, `invalidate()`, `is_expanded()`, `moved()`, `node()`, `picked_up()`, `reveal()`, `right_clicked()`, `row()`, `set_expanded()`, `set_tree()`, `sync()`, `visible()`, `with_tree()` |
| `engine-render::ui::tree_widget` | [tree_widget.md](tree_widget.md) | 573 | `Source`, `Tree`, `edit()`, `editing()`, `grab()`, `invalidate()`, `is_expanded()`, `node()`, `on_move()`, `on_pick_up()`, `on_row_click()`, `on_row_double_click()`, `on_row_drop()`, `on_row_right_click()`, `on_select()`, `reveal()`, `row()`, `select()`, `selected()`, `set_expanded()`, `show()`, `tagged()` |
| `engine-render::ui::viewport` | [viewport.md](viewport.md) | 140 | `Viewport`, `camera()`, `node()`, `set_camera()` |
| `engine-render::ui::widget` | [widget.md](widget.md) | 985 | `Button`, `ButtonStyle`, `Checkbox`, `CheckboxStyle`, `Label`, `Menu`, `Popup`, `Progress`, `ProgressStyle`, `RadioGroup`, `RadioStyle`, `Scrollbar`, `ScrollbarStyle`, `Slider`, `SliderStyle`, `StateStyle`, `TabStyle`, `Tabs`, `TextField`, `button()`, `checkbox()`, `checked()`, `fills()`, `node()`, +15 more |

[engine index](../../index.md)
