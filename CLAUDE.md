on a significant change, update the `docs/notes/` note that owns the area and
the `Readme.md` row that links it — see **Documentation** below

follow YAGNI principles, and prefer one-liner solutions

avoid fallbacks as crutch for a new feature not working especially in the case
of one implementation replacing another. do not leave the previous impl in
place with the new impl silently falling back to it.

do implement fallbacks in the case of user friendliness. the program shouldn't
crash in the case a user drag and drops an incorrect value to a drop zone.

once completed, explain your changes and how they work. that explanation is a
blurb at the end of your turn — it does not go in the code.

## Reading this codebase

`crates/` is ~44k lines and reading it to find out what exists is the main way
a session runs out of context. Escalate in this order and stop as soon as you
can act:

1. **`docs/api/index.md`** — a symbol table over one generated digest per
   module. Find the symbol, open that one digest. The whole engine's public
   API is ~12k tokens here against ~380k of source. `make api` regenerates it;
   the pre-commit hook does it for you when you touch a `.rs` file.
2. **`docs/notes/<area>.md`** — the *why*. `Readme.md`'s tables link the right
   note from the row that mentions the subsystem.
3. **the LSP tool** — go-to-definition and document symbols read one item, not
   one file. Prefer it over opening a file to look up a single signature.
4. **`tools/poke tree` / `poke focus`** — for "what is on screen right now",
   ask the running app rather than reading the code that builds the UI.
5. **the source of the one file you are changing** — and only that one.

Use the **Explore subagent** for open-ended "where is X handled" or "what
calls Y" questions across more than two files. It reads in its own context and
returns the `file:line`, so the search never lands in this one. Do not use it
for a question `docs/api/index.md` already answers.

Never read `crates/engine-render/src/ui/` whole. It is 13k lines, and
[ui-core](docs/notes/ui-core.md), [ui-widgets](docs/notes/ui-widgets.md) and
[ui-widget-authoring](docs/notes/ui-widget-authoring.md) cover it.

## Session cost

what a session costs is **turns times the context they carry**, not the bytes
of any one read. so:

- batch independent reads, greps and checks into one block. ten `sed -n`
  calls cost ten times what one block of ten does.
- never re-read a file you just edited, and never `git diff` your own work to
  see whether the edit landed. the edit tools fail loudly.
- rewrap prose with `tools/wrap <file.md>` — one call. a read-fix-check loop
  over line lengths is the most expensive way to move a word.
- one `locate` spawn beats four greps here; one `verify-app` spawn beats
  building, driving and screenshotting here. both are Sonnet.
- reach for the LSP before opening a file to read a single signature.

## Documentation

`Readme.md` is the map: what exists, where it lives, and a link to the note
with the reasoning. Keep it that way — **a Readme row is at most two or three
sentences**. Anything longer is a design narrative and belongs in
`docs/notes/`, which is also the only way it stays true, because a note is
owned by one topic and gets updated when that topic changes.

Do not add signatures or API listings to prose; `docs/api/` is generated and
cannot drift.

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
they predate the rule. **do not imitate them, and do not rewrite them
unasked.**

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
tools/poke rclick "cube"         # secondary button -> context menu
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
