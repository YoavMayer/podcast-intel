# State machines

Eight implicit state machines, found 11.11.2026. Three are now explicit.

## 1. Episode transcription status (explicit)

Column `episodes.transcription_status`. Table in `models/state.py`, enforced by `Database.transition_episode_status`.

| State | Next | Set in |
|---|---|---|
| pending | processing | schema default |
| processing | completed, failed, pending | both transcribers |
| failed | processing, pending | `whisper_transcriber.py` on error |
| completed | pending | transcribers |

Fixed: `failed` was never written, so a crash left `processing` forever. Re-running a completed episode duplicated segments; it is now rejected.

## 2. Audio download marker (explicit)

"File exists" means "downloaded" in `whisper_transcriber.download_episode`, `tools/run_episode_analysis.download_audio` and `tools/diarize_episode.download_audio`. States: absent, partial, complete.

Fixed: an interrupted download left a partial file read as complete. Now written to `.part`, then renamed.

## 3. Transcript marker (explicit)

`reports/episode_N/transcript.json` existing skips transcription (`run_episode_analysis.py`) and lists episodes in two diarization tools. Fixed the same way. Open: nothing ties it to the audio or model that made it.

## Documented, not changed

| Machine | States | Where | Risk |
|---|---|---|---|
| 4. Pipeline steps | rss, register, download, transcribe, diarize, metrics, reports; each ok, skipped, error, timeout or exception | `run_full_pipeline` | Order is code order. Reports run even when metrics failed. Results are saved only at the end, so a crash or early return leaves no record. |
| 5. RSS discovery | unknown, known | `rss_watcher.load_known_guids` | Known means registered, not processed, so a failed episode is never picked up again. |
| 6. RSS poll | waiting (attempt N), found, exhausted | `poll_rss_for_episode` | Errors are swallowed per attempt. |
| 7. Trigger state | run_count, last_run, last_error | `TriggerState`, `load_triggers` | Never read. `enabled` is ignored, pipeline names unchecked. |
| 8. Event status | SCHEDULED, TIMED, IN_PLAY, PAUSED, HALFTIME, FINISHED, POSTPONED, CANCELLED, SUSPENDED | `football.py`, `briefing_generator.py`, `cli.py` | External values matched in four places with different sets. `POSTPONED` and `CANCELLED` become type `match`. |
