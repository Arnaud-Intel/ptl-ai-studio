# Street videos

What the Smart City demo plays from disk, so that it runs with no network.
The files are **not in the repository**: each is fetched from where its
author published it, into this folder, by any of

- the first-launch helper (`first_launch.bat`), which offers it;
- **Prepare models** in the app, or `uv run panther-lake-prefetch street-videos`;
- the demo itself, the first time one of them is started.

The list, with each file's size and checksum, is
[`core/src/pantherlake_ai_core/sample_videos.py`](../../core/src/pantherlake_ai_core/sample_videos.py).
A file that is not, byte for byte, the one named there is refused.

## Licences and credits

Keep this page with the files wherever they are copied or hosted: three of
the four licences ask for the credit.

| File | What | Licence | Credit | Source |
| --- | --- | --- | --- | --- |
| `toronto-yonge-dundas-crossing.webm` | The scramble crossing at Yonge and Dundas, Toronto, at street level. 57 s, 1080p, 36.7 MB | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (no conditions) | Raysonho @ Open Grid Scheduler / Grid Engine | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm) |
| `tyumen-crossing.webm` | The crossing of Respubliki and Ordzhonikidze streets, Tyumen, at street level. 31 s, 1080p, 10.4 MB | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | RG72 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kruci%C4%9Do_de_stratoj_Respubliko_kaj_Or%C4%9Donikidze_(Tjumeno).webm) |
| `tokyo-shibuya-crossing.webm` | Shibuya Crossing, Tokyo, from above. 59 s, 1080p, 37.4 MB | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Basile Morin | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm) |
| `intel-person-bicycle-car.mp4` | People, bicycles and cars on a quiet street. 54 s, 768x432, 6.0 MB | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Intel Corporation | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) |

Toronto and Shibuya are the 1080p versions Wikimedia Commons itself makes of
the original uploads, unchanged; Tyumen is the original upload. They show
people in public streets, as any city camera does; nothing is known or said
about who they are.

The demo opens on the first two, which are the two its detector does best
on; the table of what was measured is in `sample_videos.py`.
