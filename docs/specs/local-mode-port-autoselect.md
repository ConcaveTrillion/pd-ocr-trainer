# Spec: local-mode port auto-select + stable bookmarks + in-UI URL display

**Status:** spec only, not implemented.
**Owner repo:** `pd-ocr-trainer` (NiceGUI training UI).
**Workspace-wide:** sibling specs land in every local-web-app pd-* repo
— `pd-prep-for-pgdp` (commit `b23b913`), `pd-ocr-labeler-spa`
(commit `b956275`), and `pd-ocr-labeler` (sibling item being added in
parallel). The behaviors below MUST converge across all four so
operators get one predictable dev-UX.

## Motivation

Triggered by a real stale-port collision: a previously-launched local
NiceGUI training UI held the default port, and the next launch failed
loudly without a fallback. Local-mode operators are not running a
production deployment — they want the UI to come up, not to
hand-debug port collisions.

Trainer's NiceGUI UI is also long-running (training oversight is the
whole point), so operators routinely close the launching console and
leave only the browser tab open. If the bound port shifts between
restarts, bookmarks break silently. The URL must therefore also be
visible from inside the UI itself, not only on stdout.

## Proposed behavior — local mode only

This spec applies **only** when the UI is started in local/dev mode
(the `make run` / `make run-verbose` path). Any future hosted /
shared-deployment mode must use an explicit, fixed port and is out of
scope.

1. **Auto-select fallback.** Port resolution order on startup:
   1. Persisted-last-port (see Stable bookmarks below), if present and
      free.
   2. Default port (current hardcoded NiceGUI port for the trainer UI).
   3. `port=0` — kernel picks a free ephemeral port.

   If the operator passes `--port N` explicitly, the auto-select
   fallback is **disabled**: collision must fail loud with a clear
   message naming the conflicting port. Explicit intent is honored.

2. **Stable bookmarks.** Persist the last successfully-bound port to a
   small state file under the workspace state dir (e.g.
   `~/.config/pd-ocr-trainer/last-port` or the equivalent
   `platformdirs` location — exact path TBD at implementation time).
   Read on next start; on read-failure, fall through to default port
   per step 1.

3. **URL visible in the running UI.** The NiceGUI page must surface
   the current bound URL (scheme + host + port) somewhere persistent
   that survives navigation between training screens. Acceptable
   placements: footer strip, header right-side, About panel, or a
   copy-to-clipboard widget. Belt-and-suspenders: keep the existing
   stdout banner *and* add the in-UI display — operators who closed
   the console still need to find the URL.

## Acceptance criteria

Tests required (all local-mode):

- `default-port-free` — default port available; UI binds default port;
  `last-port` state file written.
- `default-port-taken-fallback-succeeds` — default port held by another
  process; UI binds via `port=0`; `last-port` state updated to the new
  ephemeral port; no error.
- `explicit-port-collision-fails-loud` — `--port N` passed and `N` is
  taken; UI exits non-zero with a message naming `N`; no fallback
  attempted; `last-port` not modified.
- `persisted-port-reused-on-next-start` — prior run wrote
  `last-port=X`; `X` is free; UI binds `X` (not the default).
- `persisted-port-taken-falls-through` — `last-port=X`; `X` is taken;
  UI falls through to default, then `port=0` per step 1.
- `in-ui-url-visible` — page-render assertion that the current
  scheme+host+port appears in the served HTML (footer / header /
  About / copy widget — whichever placement implementation chooses).

## Cross-repo links

- `pd-prep-for-pgdp`: commit `b23b913` — sibling spec entry.
- `pd-ocr-labeler-spa`: commit `b956275` — sibling spec entry.
- `pd-ocr-labeler`: sibling roadmap item being added in parallel
  (legacy NiceGUI labeler, same behavior surface as this trainer UI).

## Out of scope

- Any hosted / multi-user deployment mode of the trainer UI.
- Auto-killing the process holding the default port (collisions are
  signaled, never resolved by force).
- Browser-side bookmark rewriting; the bookmark stability story is
  "the URL stays put across restarts," not "we update existing
  bookmarks."
