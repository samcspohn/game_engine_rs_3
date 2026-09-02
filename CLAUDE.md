update Readme.md with current implementation details if there are any significant changes

follow YAGNI principles, and prefer one-liner solutions

avoid fallbacks as crutch for a new feature not working especially in the case of one implementation replacing another. do not leave the previous impl in place with the new impl silently falling back to it.

do implement fallbacks in the case of user friendliness. the program shouldn't crash in the case a user drag and drops an incorrect value to a drop zone. 

once completed, explain your changes and how they work. that explanation is a
blurb at the end of your turn — it does not go in the code.

## Comments

prefer to make code self documenting. these limits cover doc comments too, not
just `//`:

- `//` inline, and `///` on anything — struct, field, function, const, impl:
  **0 lines, 3 at the very most**
- `//!` module header at the top of a file: **under 20 lines**
- anything that outgrows those goes in `docs/notes/` or `Readme.md` and gets
  linked, not squeezed into the source

the existing code is longer than this. `ui/dock.rs`, `ui/tree.rs` and
`scene.rs` open with 30+ line headers and carry 7+ line doc comments on items.
they predate the rule. **do not imitate them, and do not rewrite them unasked.**

keep the one thing a reader cannot derive from the code. drop the rest — what
the code plainly says, and the design narrative, which is what goes stale:

```rust
// bad: narrates the body, then explains a design that will drift
/// Publish this frame's box: the camera is resized to it before the next
/// frame is recorded, and a pointer inside it belongs to the scene rather
/// than to the UI. Every frame rather than on a change — a divider drag, a
/// window resize and the panel being dragged to another edge all move it,
/// and none of them are events a camera could subscribe to.
pub fn update(&self, ui: &UiCore)

// good: the non-obvious why, and nothing else
/// A zero box means "present but not showing" — the camera then holds its
/// size, which is not what no viewport at all means.
pub fn update(&self, ui: &UiCore)
```

## Commit messages

a lowercase imperative subject line, then bullets — one per change, in the
imperative, each saying what it does and the *why* only when the why is not
obvious from the what:

```
overhaul document editing

- give each document its own sub-dock: Hierarchy, Scene and Inspector are
  sub-panels of the document panel, not of the editor
- make EntityRef carry its world id, so a row dragged into another
  document's tree is refused rather than re-parenting the wrong slot
- update Readme, ui-widgets and editor-document-split
```

no prose paragraphs, and no restating the diff line by line. a bullet that
needs more than three wrapped lines is a design note — it belongs in
`docs/notes/` or `Readme.md`, same as a comment that outgrows its budget.

## Driving the running app

Use `tools/poke`, never ydotool/kdotool/xdotool/spectacle. Start the app with
`ENGINE_DEBUG_INPUT=1` and it opens a unix socket:

```sh
ENGINE_DEBUG_INPUT=1 cargo run -p editor &
tools/poke tree                  # visible text nodes + on-screen rects
tools/poke dblclick "cube"       # aim by text, not by pixel
tools/poke wait                  # queue drained and a frame ran
tools/poke type hull; tools/poke key Enter
tools/poke focus                 # what holds the keyboard, and its text
tools/poke shot                  # PNG of the frame -> target/shots/, path printed
tools/poke rec 26 drag a --to b  # film a gesture, one PNG per frame
tools/poke quit
```

Moves are swept one interpolated step per frame, not teleported, and a white
dot marks the pointer in the window and in captures.

Input is injected into `Input` in window coordinates, so it cannot leak into
another window or race the compositor's focus. `poke shot` captures the
composited swapchain image — do not take full-screen screenshots and crop.

Grammar lives in `crates/engine-render/src/debug_input.rs`; `tools/poke` is a
dumb pipe.
