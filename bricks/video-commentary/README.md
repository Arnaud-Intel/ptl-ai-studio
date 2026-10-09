# video-commentary

**Experimental.** A model watches a video and says what is happening, a line
every few seconds, in the mood you pick.

Two models, each on its own chip, neither of them new to the studio:

1. **It sees.** A vision-language model (Qwen2.5-VL 7B, the one
   [`screen-ocr`](../screen-ocr/README.md) reads text with) is shown one
   frame and asked for one plain sentence about what is happening.
2. **It says.** A small language model (Qwen2.5-1.5B, from
   [`doc-qa`](../doc-qa/README.md)) says that sentence again in a mood:
   upbeat, a sports commentator, a nature documentary, deadpan -- or not at
   all, and the plain sentence is the comment.

The mood can change while the video plays: the last sentence is said again
in the new voice at once, without the picture being looked at again.

## What was measured

On the XPS 14 (Core Ultra X7 358H), 2026-10-09, the vision model on the
integrated GPU and the language model on the NPU:

| | |
| --- | --- |
| A plain line for one frame, 672 pixels wide | 0.8 to 1.0 s (25 tok/s, the first word after 0.2 s) |
| The same frame at 448 / 896 / 1280 wide | 0.8 / 1.0 / 1.3 s, and no better read |
| Its mood | 0.5 to 0.8 s more |
| In the launcher, first line after Start | 14 s (two models to load) |

So a comment is about a second and a half behind the picture, and a line
every four seconds leaves time to read it.

Why two models: asked for the mood directly, the vision model wrote the same
plain sentence with "bustling" in it. The small one changes the voice for
real.

## What it does not do well

- **In a mood, it embroiders.** Told to keep every fact and add none, the
  small model still turned a rider into "brave cowboys" and sent a herd "on
  their way to market". The plain line is kept beside every line in a mood,
  in the launcher and on the command line, for that reason.
- **It is a commentator, not a detector.** It looks at one frame every few
  seconds; what happens between two looks is shown and never seen.
- **"Deadpan" is barely a voice** with a model this small.

## What it does not do on purpose

No mood passes judgement on the people in the picture -- how they look, what
they wear. A commentator on a street is one thing; a machine rating the
passers-by at a stand is another: it drifts from clothes to bodies and age,
and a camera that judges visitors is not one that counts them. If that is
ever built it is its own thing -- asked for by the person, on one still,
about clothes and colours only, kind only, nothing kept -- and not one more
line in [`moods.py`](src/video_commentary/moods.py).

## Usage

OpenVINO engine only (there is no portable vision-language model here).

```bash
uv sync --extra openvino
uv run video-commentary                      # the first sample video, upbeat
uv run video-commentary --path street.mp4 --mood sports
uv run video-commentary --source webcam --mood documentary
```

| Flag | Description |
| --- | --- |
| `--source {file,webcam,screen}` | What to watch. Default: `file`. |
| `--path FILE` | The video, for `--source file`. Default: the first sample video, fetched if need be. |
| `--mood {plain,upbeat,sports,documentary,deadpan}` | The voice. Default: `upbeat`. |
| `--every SECONDS` | Seconds between two looks. Default: 4. |
| `--vision-device NAME` | Where the vision model runs. Default: a GPU. It does not compile for the NPU on this hardware. |
| `--mood-device NAME` | Where the language model runs. Default: the NPU if there is one. |
| `--once` | Play the file once instead of in a loop. |

## How it works

- [`pipeline.py`](src/video_commentary/pipeline.py) -- the video plays at
  its own pace on one thread; the commentary takes the newest frame whenever
  it is ready for one. `Commentator` decides whether there is something new
  to say: a picture that has not moved is not looked at again for twenty
  seconds, and one that moved but shows the same thing is not said twice.
- [`moods.py`](src/video_commentary/moods.py) -- a mood is an instruction to
  the language model, nothing more.
- [`samples.py`](src/video_commentary/samples.py) -- the studio's sample
  videos (see [`sample-data/videos`](../../sample-data/videos/README.md)).
