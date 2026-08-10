update Readme.md with current implementation details if there are any significant changes

follow YAGNI principles, and prefer one-liner solutions

avoid fallbacks as crutch for a new feature not working especially in the case of one implementation replacing another. do not leave the previous impl in place with the new impl silently falling back to it.

do implement fallbacks in the case of user friendliness. the program shouldn't crash in the case a user drag and drops an incorrect value to a drop zone. 

avoid overly verbose comments. prefer to make code self documenting. if a comment is needed, make it concise and to the point. prefer 1 line comments up to 3 lines

once completed, explain your changes and how they work

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
