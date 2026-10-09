"""The Auto Demo's playlist: its scenes, each built for the stand it plays on.

A scene is what the director does (which routes, with what, waiting for
what) and the story the stage tells meanwhile: a sentence or two at a time,
about what is happening, why it is done that way and how -- which chip,
which model. The story is the reason the mode exists: a stand is watched by
people nobody is there to talk to.

What goes in was decided with the user on 2026-10-08 and -09:

- The discrete GPU is not always there. Every scene says what it does with
  one and without, and the person starting the loop can leave it out.
- No sound: a stand at a large event is too loud for it.
- The story is told in English unless French is chosen when the loop is
  started. Every sentence here is written in both.
- Smart City plays only when the internet is reachable.
- "Seeing and answering" (object detection with document Q&A) is written
  and held back until both bricks have had a proofing pass (Document Q&A
  had its own on 2026-10-09): it does not play until its entry is taken
  out of `HELD_BACK`.

Writing a beat: one or two sentences, plain words, no figure that is not on
screen. A beat with a `stage` is shown when that stage of the demo is seen
at work; say there what the stage does, so the sentence and the screen
agree whatever the day's loading times are. A sentence that points at
something the stage puts out ("that is the code appearing") takes
`figure=True`: at work begins with a model loading, and the code does not.
"""
from __future__ import annotations

from .autodemo import Ask, Beat, Builder, Chip, Scene, Skip, Stand, Start, Until, Wait

# Scenes that are written but not to be played yet, and why.
HELD_BACK: dict[str, str] = {
    # Document Q&A had its pass on 2026-10-09 (the search, the prompt, the
    # index kept after the folder changed). Object Detection has not, and
    # the stage has no view yet for a camera picture beside an answer.
    "seeing-and-answering": "held back until Object Detection has had its proofing pass and the stage has a view for it",
}

_IGPU, _DGPU, _NPU, _CPU = "Integrated GPU", "Arc Pro B60", "NPU", "CPU"


def _say(stand: Stand, english: str, french: str) -> str:
    return french if stand.lang == "fr" else english


def _sample(stand: Stand, demo: str, turn: int = 0, where=lambda sample: True) -> dict | None:
    """One of a demo's bundled samples: the `turn`-th of those `where` keeps,
    going round."""
    kept = [sample for sample in stand.samples(demo) if where(sample)]
    return kept[turn % len(kept)] if kept else None


def page_agent(stand: Stand, loop: int) -> Scene | Skip:
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("Three models, three chips, one web page", "Trois modèles, trois puces, une page web")
    if not stand.igpu:
        return Skip(title, "needs a GPU for the image model and the coding model")
    sample = _sample(stand, "page-agent", loop - 1)
    if sample is None:
        return Skip(title, "its sample requests could not be read")
    two = stand.dgpu is not None
    planner_chip = _NPU if stand.npu else _IGPU
    beats = [
        Beat(say(
            f"Someone asks for a web page: \"{sample['name']}\". Three AI models are about to build it together, "
            "right here on this laptop. Nothing is sent to the cloud.",
            f"Quelqu'un demande une page web : « {sample['name']} ». Trois modèles d'IA vont la construire ensemble, "
            "ici même, sur ce portable. Rien n'est envoyé dans le cloud.",
        )),
        Beat(say(
            "First, a small model plans the page: a name, a headline, what is on offer, and six photographs to take. "
            + ("It runs on the NPU, a chip made for small jobs at a few watts."
               if stand.npu else "This machine has no NPU, so it shares the graphics chip."),
            "D'abord, un petit modèle conçoit la page : un nom, un titre, ce qu'elle propose et six photos à prendre. "
            + ("Il tourne sur le NPU, une puce faite pour les petites tâches, à quelques watts."
               if stand.npu else "Cette machine n'a pas de NPU : il partage donc la puce graphique."),
        ), stage="plan"),
        Beat(say(
            "Now an image model takes those photographs, one every few seconds, on the integrated GPU: the graphics "
            "chip inside the processor. No graphics card is needed for this.",
            "Un modèle d'image prend maintenant ces photos, une toutes les quelques secondes, sur le GPU intégré : "
            "la puce graphique du processeur. Aucune carte graphique n'est nécessaire.",
        ), stage="images"),
    ]
    if two:
        beats.append(Beat(say(
            "At the same time, a 30-billion-parameter coding model writes the page on the Arc Pro B60. "
            "Two GPUs, two jobs at once: the page is written while its pictures are still being drawn.",
            "En même temps, un modèle de code de 30 milliards de paramètres écrit la page sur l'Arc Pro B60. "
            "Deux GPU, deux tâches à la fois : la page s'écrit pendant que ses images se dessinent.",
        ), stage="page"))
    else:
        beats.append(Beat(say(
            "The pictures are done. The same chip now unloads the image model, loads a 30-billion-parameter coding "
            "model and writes the page: one GPU doing both jobs, in turn, with no graphics card.",
            "Les images sont prêtes. La même puce décharge le modèle d'image, charge un modèle de code de "
            "30 milliards de paramètres et écrit la page : un seul GPU pour les deux tâches, tour à tour.",
        ), stage="page"))
    beats += [
        Beat(say(
            "That is the page's code appearing as it is written. On the right, each chip shows its own speed: "
            "tokens per second for the models that write, images per minute for the one that draws.",
            "C'est le code de la page qui apparaît à mesure qu'il s'écrit. À droite, chaque puce affiche sa vitesse : "
            "des tokens par seconde pour les modèles qui écrivent, des images par minute pour celui qui dessine.",
        ), stage="page", figure=True),
        Beat(say(
            "And here it is: a complete, illustrated web page from a single request. Every picture and every line "
            "of it was generated on this machine a moment ago.",
            "Et la voici : une page web complète et illustrée, à partir d'une seule demande. Chaque image et chaque "
            "ligne ont été générées sur cette machine il y a un instant.",
        ), result=True),
    ]
    planner = Chip(planner_chip, say("Plans the page · Qwen3-8B", "Conçoit la page · Qwen3-8B"), "page-agent", ("plan",))
    if two:
        chips = (
            planner,
            Chip(_IGPU, say("Draws the pictures · FLUX.1-schnell", "Dessine les images · FLUX.1-schnell"), "page-agent", ("images",)),
            Chip(_DGPU, say("Writes the page · Qwen3-Coder 30B", "Écrit la page · Qwen3-Coder 30B"), "page-agent", ("page",)),
            Chip(_CPU, say("Conducts and checks · plain code", "Dirige et vérifie · du simple code")),
        )
    else:
        chips = (
            planner,
            Chip(_IGPU, say("Draws, then writes · FLUX.1 then Qwen3-Coder 30B", "Dessine puis écrit · FLUX.1 puis Qwen3-Coder 30B"),
                 "page-agent", ("images", "page")),
            Chip(_CPU, say("Conducts and checks · plain code", "Dirige et vérifie · du simple code")),
        )
    devices = {} if two else {"image_device": stand.igpu, "page_device": stand.igpu}
    return Scene(
        id="page-agent",
        title=title,
        demo="page-agent",
        view="page",
        beats=tuple(beats),
        chips=chips,
        props={"request": sample["name"]},
        steps=(Ask("/api/page-agent/build", {"request": sample["prompt"], **devices},
                   cancel="/api/bricks/page-agent/stop?stage=page"),),
        hold=40.0,
        at_most=480.0 if two else 780.0,
    )


def expense_extraction(stand: Stand, loop: int) -> Scene | Skip:
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("Two chips share one job", "Deux puces pour un même travail")
    if not stand.igpu:
        return Skip(title, "needs a GPU to read the receipts")
    # The worn ones -- faded print, folds, shadows -- as the user asked on
    # 2026-10-09: receipts that look like receipts, not like a form.
    sample = _sample(stand, "expense-extract", 0, lambda s: s.get("name", "").startswith("Scanned") and s.get("folder"))
    if sample is None:
        return Skip(title, "its sample receipts could not be found")
    structuring = "NPU" if stand.npu else stand.igpu
    receipts = [{"name": asset["name"], "url": asset["url"]} for asset in sample.get("assets") or [] if asset.get("image")]
    return Scene(
        id="expense-extraction",
        title=title,
        demo="expense-extract",
        view="receipts",
        beats=(
            Beat(say(
                "Five worn receipts, with faded print, folds and shadows, have to become an expense report. "
                "Usually that is somebody typing. Here, two AI models share the work.",
                "Cinq reçus fatigués, à l'encre pâlie, pliés, mal éclairés, doivent devenir une note de frais. "
                "D'habitude, quelqu'un les saisit à la main. Ici, deux modèles d'IA se partagent le travail.",
            )),
            Beat(say(
                "A vision model reads each receipt on the integrated GPU. Reading a photograph is the heavy half of "
                "the job, so it gets the graphics chip.",
                "Un modèle de vision lit chaque reçu sur le GPU intégré. Lire une photo est la partie lourde du "
                "travail : elle revient donc à la puce graphique.",
            ), stage="ocr", after=4.0),
            Beat(say(
                "As soon as a receipt is read, a small language model turns its text into a clean line: vendor, date, "
                "amount, category. " + ("It runs on the NPU, so the GPU never has to stop reading."
                                         if stand.npu else "On this machine it shares the GPU."),
                "Dès qu'un reçu est lu, un petit modèle de langage en fait une ligne propre : fournisseur, date, "
                "montant, catégorie. " + ("Il tourne sur le NPU : le GPU n'a jamais à s'arrêter de lire."
                                          if stand.npu else "Sur cette machine, il partage le GPU."),
            ), stage="llm", after=14.0),
            Beat(say(
                "Receipts in, expense lines out. Where a figure cannot be matched to what is printed, a total too "
                "faded to read for instance, the line is flagged for a person to check instead of being trusted.",
                "Des reçus en entrée, des lignes de frais en sortie. Quand un chiffre ne se retrouve pas sur le reçu, "
                "un total trop pâle pour être lu par exemple, la ligne est signalée à une personne au lieu d'être crue.",
            ), result=True),
        ),
        chips=(
            Chip(_IGPU, say("Reads the receipts · Qwen2.5-VL 7B", "Lit les reçus · Qwen2.5-VL 7B"), "expense-extract", ("ocr",)),
            Chip(_NPU if stand.npu else _IGPU, say("Fills in the lines · Qwen2.5 1.5B", "Remplit les lignes · Qwen2.5 1.5B"),
                 "expense-extract", ("llm",)),
        ),
        props={"receipts": receipts},
        steps=(
            Start("/api/expense-extract/report/close"),
            Start("/api/expense-extract/start", {
                "folder": sample["folder"], "ocr_engine": "openvino", "ocr_compute_device": stand.igpu,
                "llm_engine": "openvino", "llm_compute_device": structuring,
            }),
            Wait(3.0),
            Until("/api/expense-extract/report", "running", False, timeout=300.0),
        ),
        stop=("/api/expense-extract/stop", "/api/expense-extract/report/close"),
        hold=30.0,
        at_most=360.0,
    )


def smart_city(stand: Stand, loop: int) -> Scene | Skip:
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("Street cameras, one per chip", "Des caméras de rue, une par puce")
    if not stand.igpu:
        return Skip(title, "needs a GPU")
    single = [s for s in stand.samples("smart-city-monitor")
              if "|" not in str(s.get("feeds", "")) and "\n" not in str(s.get("feeds", ""))]
    # The street videos kept on this machine, when they have been fetched:
    # full pictures, people as well as cars, and no network to depend on.
    # The first two, which are the two in high definition; each turn of the
    # loop they change chips.
    on_disk = [s for s in single if s.get("kind") == "file" and s.get("ready")][:2]
    from_disk = len(on_disk) == 2
    if from_disk:
        first, second = on_disk if loop % 2 else reversed(on_disk)
    else:
        if not stand.internet:
            return Skip(title, "needs its street videos fetched (Prepare models), or the internet for live cameras")
        # The traffic-camera clips: plain video files, where the live streams
        # need a video site to let the laptop in.
        clips = [s for s in single if s.get("kind") != "file" and s.get("group") != "YouTube"]
        if len(clips) < 2:
            return Skip(title, "fewer than two street-camera sources to choose from")
        first, second = clips[(loop - 1) % len(clips)], clips[loop % len(clips)]
    other = "NPU" if stand.npu else "CPU"
    other_chip = _NPU if stand.npu else _CPU
    return Scene(
        id="smart-city",
        title=title,
        demo="smart-city-monitor",
        view="cameras",
        beats=(
            Beat(say(
                "Two street scenes, watched at the same time the way a city watches its cameras. "
                "The videos play from this laptop's disk: nothing here needs a network.",
                "Deux scènes de rue, surveillées en même temps, comme une ville surveille ses caméras. "
                "Les vidéos sont lues depuis le disque de ce portable : rien ici n'a besoin du réseau.",
            ) if from_disk else say(
                "Two traffic cameras in London, watched at the same time. The pictures come from the city; "
                "everything that happens to them happens on this laptop.",
                "Deux caméras de circulation à Londres, surveillées en même temps. Les images viennent de la ville ; "
                "tout ce qui leur arrive ensuite se passe sur ce portable.",
            )),
            Beat(say(
                # "Looks for", not "finds every": in a crowd seen from far
                # above, this detector boxes a handful of several hundred.
                "A detector looks for people and vehicles in every frame, and follows each one it finds. For the "
                "first camera it runs on the integrated GPU, the fastest chip for many small jobs a second.",
                "Un détecteur cherche les personnes et les véhicules dans chaque image, et suit chacun de ceux qu'il "
                "trouve. Pour la première caméra, il tourne sur le GPU intégré, la puce la plus rapide pour "
                "beaucoup de petites tâches par seconde.",
            ), stage="feed-1", after=8.0),
            Beat(say(
                f"The second camera has its own chip: the {other_chip}. " + (
                    "That is what an NPU is for: one small model, all day, at a few watts."
                    if stand.npu else "This machine has no NPU, so the processor takes it."),
                f"La seconde caméra a sa propre puce : le {other_chip}. " + (
                    "C'est à cela que sert un NPU : un petit modèle, toute la journée, à quelques watts."
                    if stand.npu else "Cette machine n'a pas de NPU : le processeur s'en charge."),
            ), stage="feed-2", after=24.0),
            Beat(say(
                "Look at the two frame rates on the right: neither drops because the other is working. "
                "And everything is counted as it crosses the picture, without a frame leaving the machine.",
                "Regardez les deux cadences à droite : aucune ne baisse parce que l'autre travaille. "
                "Et tout est compté au passage, sans qu'une seule image ne quitte la machine.",
            ), after=44.0),
        ),
        chips=(
            Chip(_IGPU, say("Camera 1 · YOLO11s detector", "Caméra 1 · détecteur YOLO11s"), "smart-city-monitor", ("feed-1",)),
            Chip(other_chip, say("Camera 2 · YOLO11s detector", "Caméra 2 · détecteur YOLO11s"), "smart-city-monitor", ("feed-2",)),
        ),
        props={"feeds": [{"id": "feed-1", "name": first["name"], "chip": _IGPU},
                         {"id": "feed-2", "name": second["name"], "chip": other_chip}]},
        steps=(
            Start("/api/smart-city-monitor/start", {
                "engine": "openvino", "loop": True,
                "feeds": [{"path": first["feeds"], "compute_device": stand.igpu},
                          {"path": second["feeds"], "compute_device": other}],
            }),
            Wait(70.0),
        ),
        stop=("/api/smart-city-monitor/stop",),
        hold=0.0,
        at_most=150.0,
    )


def seeing_and_answering(stand: Stand, loop: int) -> Scene | Skip:
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("Seeing and answering at once", "Voir et répondre en même temps")
    if "seeing-and-answering" in HELD_BACK:
        return Skip(title, HELD_BACK["seeing-and-answering"])
    if not (stand.igpu and stand.npu):
        return Skip(title, "needs a GPU and an NPU")
    sample = _sample(stand, "doc-qa", loop - 1, lambda s: s.get("folder") and s.get("question"))
    if sample is None:
        return Skip(title, "its sample documents could not be found")
    camera = bool(stand.cameras)
    source = {"source": "camera", "camera_index": stand.cameras[0]} if camera else {"source": "screen"}
    return Scene(
        id="seeing-and-answering",
        title=title,
        demo="doc-qa",
        view="answer",
        beats=(
            Beat(say(
                ("A camera on this stand is being watched by an object detector. Nothing it sees is recorded. "
                 if camera else "The screen is being watched by an object detector. ")
                + "It runs on the integrated GPU.",
                ("Une caméra de ce stand est surveillée par un détecteur d'objets. Rien de ce qu'elle voit n'est "
                 "enregistré. " if camera else "L'écran est surveillé par un détecteur d'objets. ")
                + "Il tourne sur le GPU intégré.",
            )),
            Beat(say(
                "Meanwhile, on the NPU, a language model reads a folder of business documents and answers a question "
                "about them. Two unrelated jobs, two chips, at the same time.",
                "Pendant ce temps, sur le NPU, un modèle de langage lit un dossier de documents professionnels et "
                "répond à une question à leur sujet. Deux tâches sans rapport, deux puces, en même temps.",
            ), after=10.0),
            Beat(say(
                "The answer is written from the documents themselves, and says which file each part came from.",
                "La réponse est rédigée à partir des documents eux-mêmes, et indique de quel fichier vient chaque partie.",
            ), result=True),
        ),
        chips=(
            Chip(_IGPU, say("Detects objects · YOLO11s", "Détecte les objets · YOLO11s"), "object-detection"),
            Chip(_NPU, say("Answers from documents · Qwen2.5 1.5B", "Répond d'après les documents · Qwen2.5 1.5B"), "doc-qa"),
        ),
        props={"question": sample["question"]},
        steps=(
            Start("/api/object-detection/start", {**source, "engine": "openvino", "compute_device": stand.igpu}),
            Ask("/api/doc-qa/ingest", {"folder": sample["folder"], "engine": "openvino", "compute_device": "NPU"}),
            Ask("/api/doc-qa/ask", {"question": sample["question"]}, cancel="/api/bricks/doc-qa/stop"),
        ),
        stop=("/api/object-detection/stop",),
        hold=30.0,
        at_most=240.0,
    )


# In the order they play. Heavy and light alternate: the 30B model lost a
# fifth of its speed after many runs in a row (docs/AUTO_DEMO.md).
PLAYLIST: list[Builder] = [page_agent, expense_extraction, smart_city, seeing_and_answering]
