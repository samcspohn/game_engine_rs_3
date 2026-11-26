# Todo

## Gaps

- an edit can't clear a slot: an empty asset or entity is left out, so a model's value stands
- an empty asset inside a list is written as `""`, which a mesh reads back as "no mesh chosen"
- edits to an anonymous entity aren't recorded for undo
- saving names an anonymous entity `entity`, `entity1`, …
- the inspector header keeps the old name after a rename until reselected
- `test-game` doesn't build with `--profile editor` without the editor's env vars
- the release `test-game` build warns about an unused import

## Put off

- moving a resource file breaks the scenes that point at it
- resources that live inside a scene (`["#name"]`) aren't built
- scene values are held as JSON in memory
- *Make unique* for a resource isn't built
- *reload scene* doesn't re-read the scenes an instance uses
- quitting doesn't ask about unsaved changes
- the packager ships all of `assets/` and `scenes/`, used or not
- a project that is only scenes has no stock player binary to package
- no written Vulkan feature floor, and no clean failure below it
- no asset cooking: no transcode, pre-bake or pack file
