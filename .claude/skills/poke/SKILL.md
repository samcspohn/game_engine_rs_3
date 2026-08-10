---
name: poke
description: Drive and screenshot the running engine app (editor or test-game) — click, double-click, drag, type, press keys, query live UI state, and capture the frame as a PNG. Use this INSTEAD of ydotool, kdotool, xdotool, wtype or spectacle whenever you need to interact with the app or see what it looks like. Triggers - "click the button", "type into the field", "screenshot the app", "does it look right", "verify the UI", "test the widget in the real app", "drive the editor".
---

# poke — drive the running app

A unix socket inside the app. Input is addressed to *this process* in window
coordinates, so it cannot leak into another window, cannot race the
compositor's focus, and needs no window manager.

**Never use ydotool / kdotool / xdotool / wtype / spectacle for this.** They
type into whatever the compositor has focused (which has landed text in the
user's editor before), cannot position the mouse (ydotoold's virtual device
has no `EV_ABS` axes), and force full-screen screenshots that then need
cropping against monitor geometry.

## Start the app

```sh
ENGINE_DEBUG_INPUT=1 cargo run -p editor &      # or -p test-game
sleep 15                                        # first build/init is slow
```

Set `ENGINE_DEBUG_INPUT=<path>` instead of `1` to run two apps at once.
Without the variable the socket is never opened and the app is unchanged.

## Commands

| | |
|---|---|
| `tools/poke tree` | every visible text node: `<node> <x> <y> <w> <h> <text>` |
| `tools/poke find <text>` | the same, filtered by substring |
| `tools/poke click <x>,<y>` | click a coordinate |
| `tools/poke click "<text>"` | click the centre of the node showing that text |
| `tools/poke dblclick <target>` | double click |
| `tools/poke drag <target> --to <target>` | press, move, release |
| `tools/poke move <x>,<y>` / `wheel <lines>` | hover / scroll |
| `tools/poke type <text>` | insert text |
| `tools/poke key <spec>` | `Enter` `Tab` `Escape` `Backspace` `Left` `ctrl+a` `shift+End` |
| `tools/poke focus` | what holds the keyboard, and its text if it is a field |
| `tools/poke wait` | block until the queue drains and a frame runs |
| `tools/poke shot [path]` | PNG of the composited frame; prints the path |
| `tools/poke quit` | stop the app |

## How to use it

**Aim by text, not by pixel.** `find` / `click "cube"` resolve through the
solved layout with group offsets and clipping applied, so a scrolled row
reports where it actually is. Computing a coordinate by hand re-derives a
layout the engine already solved.

**`wait` after anything that changes state**, before querying. Input is queued
one step per frame — a click is `[move, press, release]` over three frames —
and `wait` returns once the queue is empty and a frame has consumed it.

**Assert with `focus` / `find`, not with screenshots.** A query is exact and
cheap; an image needs eyeballing. Use `shot` to check *appearance* — layout,
colour, whether a widget looks right — not to read back state.

```sh
tools/poke dblclick cube && tools/poke wait
tools/poke type hull && tools/poke key Enter && tools/poke wait
tools/poke find hull        # 22.0 46 71 28 11 hull   ← assertable
tools/poke shot             # …and a picture if you need to look
```

`shot` writes to `target/shots/` (override with `ENGINE_SHOT_DIR`) and prints
an absolute path — read that file directly.

## Notes

- The grammar lives in `crates/engine-render/src/debug_input.rs`; `tools/poke`
  is a dumb pipe, so extend the Rust side.
- `pkill -f "target/debug/editor"` matches its own command line and kills the
  wrapping shell — use `pkill -f "[t]arget/debug/editor"`, or `poke quit`.
- Capture is the swapchain image, so it includes the UI and is pixel-exact
  against what the compositor shows.
