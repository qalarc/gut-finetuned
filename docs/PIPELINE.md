# Video Creation Pipeline — the repeatable process

**Saved 2026-09-24. This is the full recipe used for the GUT promo (`promo/gut_promo.mp4`).
Follow it top-to-bottom for any qalarc project video. Companion: DESIGN_SYSTEM.md.**

## 0. Ground rules
1. **Script first** (owner structure): quick intro → problem → solution (brief) → how it
   works → unique pull (REAL stats) → proof (live footage) → FREE + qalarc.com. ≤90s.
2. **No rhetorical "not X, it's Y" constructions.** Plain confident selling.
3. **Every narration line must have its visual ON SCREEN at that moment** — scene list is
   locked to the VO timeline before any rendering.
4. Acronyms the VO reads must be spoken expanded ("open source intelligence", not "OSINT").
5. Subtitles: currently OFF (burned-in SRT judged inconsistent — rebuild properly later).

## 1. Script → VO → timeline
```
promo/script_v3.md             # scene table: footage | narration | on-screen text
laya-venv/bin/python promo/voiceover.py
  → vo0..vo6.mp3               # edge-tts en-AU-WilliamNeural
  → scene_durations.json       # THE timeline (VO length + 0.45s pad per scene)
  → captions.srt               # word-boundary timed (unused while subs are off)
```
Voice: `en-AU-WilliamNeural`. Edit SCENES in voiceover.py; the durations file is the
single source of truth for every downstream step.

## 2. Animated scenes — frames.html + record_browser.py
```
frames.html? f=<scene 0..6> & t=<0..1>     # __setT(t) drives every animation
laya-venv/bin/python promo/record_browser.py --animate "file://$PWD/promo/frames.html" <F> <SECS> --name <take>
```
- SECS = the scene's duration from scene_durations.json (visuals land on the VO beat).
- The recorder steps `__setT` ~15×/s and forces a screenshot per step → fluid at 30fps.
- Output: `recordings/<take>/frame_*.png` + `frame_*.json` (wall-clock ts per frame).

**Gotchas (all bit us — do not repeat):**
- headless chromium does NOT composite on JS style changes → screencast emits nothing;
  screenshot-stepping per animation step is the fix.
- concat demuxer needs explicit `duration <hold>` directives per image (plain file lists
  produce 1-frame videos / DTS errors).

## 3. Live browser footage — film-walk mode
```
record_browser.py --film-walk <seed-url> <pages> --secs <cap> [--laya http://127.0.0.1:PORT/v1/systemone] --name <take>
```
- Walks the site with the decider (Jev cloud default; `--laya` = local fine-tuned model),
  injects a **decision HUD** overlay per hop (choice, confidence, ms, "saved as training
  row") and screencasts the real viewport.
- **Thread rule:** playwright sync objects are NOT thread-safe — the walk must run INLINE
  on the main thread; only the CDP frame callback rides another thread. (The threaded
  version filmed a white page.)
- **Seed rule:** never pre-add the seed to `visited` (the walk dies instantly, 0 pages).
- Screencast ACK: send `Page.screencastFrameAck` WITHOUT sessionId for page-level
  sessions, or chromium pauses after ~3 frames.
- Verify takes with a brightness check (real pages ≈ 170-240 avg; 255 = blank).

## 4. Assembly
```
laya-venv/bin/python promo/assemble_v3.py
```
Per scene: normalize to 30fps/1920x1080/exact duration → concat → VO track (per-scene
wav + pad) → music bed (trimmed, sidechain-ducked under VO — remember `-map "[bed]"`)
→ mux. Scene→clip mapping lives in `plan()`; console/site film segments cut with
`-ss` offsets.

Music: `/home/fivelidz/Downloads/music_generated/music_general_add/selected_music/
game_open_menu_load_cycle_baseWorld 05.wav` (current). Ducked via
`sidechaincompress=threshold=0.05:ratio=8:attack=20:release=400`.

## 5. QC checklist
- duration ≤ 90s · streams = video+audio
- scene 1 visual = problem cards at "slow/hallucination" narration beat
- film segment shows REAL pages (brightness check) + HUD decisions
- no subtitles · music ducked (VO clearly on top) · stats counters animate
