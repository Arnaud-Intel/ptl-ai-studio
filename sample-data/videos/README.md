# Sample videos

What the Smart City demo plays from disk, so that it runs with no network:
four streets, and two that are not streets -- a bottle line and a herd --
for the same detector to count other things in.
The files are **not in the repository**: each is fetched from where its
author published it, into this folder, by any of

- the first-launch helper (`first_launch.bat`), which offers it;
- **Prepare models** in the app, or `uv run panther-lake-prefetch street-videos`;
- the demo itself, the first time one of them is started.

The list, with each file's size and checksum, is
[`core/src/pantherlake_ai_core/sample_videos.py`](../../core/src/pantherlake_ai_core/sample_videos.py).
A file that is not, byte for byte, the one named there is refused.

The street videos are named by their city and nothing more, in the app and
here. Exactly where each was filmed is on its source page, linked below for
the credit its licence asks for.

## Licences and credits

Keep this page with the files wherever they are copied or hosted: four of
the six licences ask for the credit.

| File | What | Licence | Credit | Source |
| --- | --- | --- | --- | --- |
| `toronto.webm` | A busy crossing in Toronto, at street level. 57 s, 1080p, 36.7 MB | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (no conditions) | Raysonho @ Open Grid Scheduler / Grid Engine | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm) |
| `tyumen.webm` | A crossing in Tyumen, at street level. 31 s, 1080p, 10.4 MB | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | RG72 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kruci%C4%9Do_de_stratoj_Respubliko_kaj_Or%C4%9Donikidze_(Tjumeno).webm) |
| `tokyo.webm` | A large crossing in Tokyo, from above. 59 s, 1080p, 37.4 MB | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Basile Morin | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm) |
| `bottle-capping-line.webm` | Glass bottles through the capping machine of a distillery in Belgium. 21 s, 1080p, 7.1 MB | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Work With Sounds / La Fonderie | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Capping_machine_in_action.webm) |
| `cattle-drive.webm` | A herd moved to its summer range along a gravel road, Oregon. 28 s, 720p, 20.7 MB | Public domain (a work of the US Bureau of Land Management) | Bureau of Land Management Oregon and Washington | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Moving_cows_to_the_summer_range_(42877666722).webm) |
| `intel-person-bicycle-car.mp4` | People, bicycles and cars on a quiet street. 54 s, 768x432, 6.0 MB | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Intel Corporation | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) |

Toronto and Tokyo are the 1080p versions Wikimedia Commons itself makes of
the original uploads, unchanged; Tyumen, the bottle line and the herd are the
original uploads. They show
people in public streets, as any city camera does; nothing is known or said
about who they are.

The demo opens on the first two, which are the two its detector does best
on; the table of what was measured is in `sample_videos.py`.
