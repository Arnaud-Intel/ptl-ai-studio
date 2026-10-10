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
- "Seeing and answering" (object detection beside document Q&A) played
  for an evening and was taken apart on 2026-10-09: the user found that nobody
  could tell what the Q&A half was showing. Its two halves are scenes of
  their own now -- the documents, asked without the files and then with
  them; and the camera, with the visitor boxed and put into a sentence.
  `HELD_BACK` stays for the next scene written before its demo is ready.
- Which scenes play is chosen when the loop is started (asked for by the
  user on 2026-10-09): every scene has a key the start screen ticks, and a
  camera is only switched on if the person starting the loop left it in.

Writing a beat: one or two sentences, plain words, no figure that is not on
screen. A beat with a `stage` is shown when that stage of the demo is seen
at work; say there what the stage does, so the sentence and the screen
agree whatever the day's loading times are. A sentence that points at
something the stage puts out ("that is the code appearing") takes
`figure=True`: at work begins with a model loading, and the code does not.
"""
from __future__ import annotations

from .autodemo import Ask, Beat, Chip, Entry, Scene, Skip, Slot, Stand, Start, Until, Wait

# Scenes that are written but not to be played yet, by their key, and why.
HELD_BACK: dict[str, str] = {}

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


def _counts(sample: dict) -> str:
    """What a one-feed sample of the city monitor counts: street traffic
    unless it says otherwise."""
    return (sample.get("counting") or ["street"])[0]


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
    on_disk = [s for s in single if s.get("kind") == "file" and s.get("ready") and _counts(s) == "street"][:2]
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


def herd_and_line(stand: Stand, loop: int) -> Scene | Skip:
    """The same brick and the same detector as the street scene, counting
    other things: the point is that nothing was trained for them."""
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("Count whatever passes", "Compter tout ce qui passe")
    if not stand.igpu:
        return Skip(title, "needs a GPU")
    on_disk = [s for s in stand.samples("smart-city-monitor")
               if s.get("kind") == "file" and s.get("ready") and "\n" not in str(s.get("feeds", ""))]
    herd = next((s for s in on_disk if _counts(s) == "herd"), None)
    line = next((s for s in on_disk if _counts(s) == "line"), None)
    if herd is None or line is None:
        return Skip(title, "needs its sample videos fetched (Prepare models)")
    other = "NPU" if stand.npu else "CPU"
    other_chip = _NPU if stand.npu else _CPU
    return Scene(
        id="herd-and-line",
        title=title,
        demo="smart-city-monitor",
        view="cameras",
        beats=(
            Beat(say(
                "The same detector, pointed at other work. On the left, cattle driven along a road; on the right, "
                "bottles on a capping line. Both videos play from this laptop's disk.",
                "Le même détecteur, face à un autre travail. À gauche, un troupeau mené sur une route ; à droite, "
                "des bouteilles sur une ligne de capsulage. Les deux vidéos sont lues depuis le disque de ce portable.",
            )),
            Beat(say(
                "It was never trained on this herd or this factory. It knows what a cow and a bottle look like, "
                "finds them in each frame and follows each one across the picture.",
                "Il n'a jamais été entraîné sur ce troupeau ni sur cette usine. Il sait à quoi ressemblent une vache "
                "et une bouteille, les trouve dans chaque image et suit chacune à travers le cadre.",
            ), stage="feed-1", after=8.0),
            Beat(say(
                f"Each video has its own chip again: the herd on the integrated GPU, the line on the {other_chip}. "
                "A pasture gate or a production line is a camera and a few watts away from being counted.",
                f"Chaque vidéo a de nouveau sa puce : le troupeau sur le GPU intégré, la ligne sur le {other_chip}. "
                "Une barrière de pâturage ou une ligne de production : une caméra et quelques watts suffisent à compter.",
            ), stage="feed-2", after=20.0),
            Beat(say(
                # Measured: about 90 "cows" counted for a herd of a few dozen.
                "The tallies under each picture rise as things pass. They run high: an animal hidden behind "
                "another and seen again is counted twice. Enough to watch a flow, not yet to bill by.",
                "Les compteurs sous chaque image montent au passage. Ils voient large : un animal caché derrière "
                "un autre puis revu est compté deux fois. Assez pour suivre un flux, pas encore pour facturer.",
            ), after=32.0),
        ),
        chips=(
            Chip(_IGPU, say("Counts the herd · YOLO11s detector", "Compte le troupeau · détecteur YOLO11s"),
                 "smart-city-monitor", ("feed-1",)),
            Chip(other_chip, say("Counts the bottles · YOLO11s detector", "Compte les bouteilles · détecteur YOLO11s"),
                 "smart-city-monitor", ("feed-2",)),
        ),
        props={"feeds": [{"id": "feed-1", "name": herd["name"], "chip": _IGPU},
                         {"id": "feed-2", "name": line["name"], "chip": other_chip}]},
        steps=(
            Start("/api/smart-city-monitor/start", {
                "engine": "openvino", "loop": True,
                # The herd on the GPU: its clip is the one at 30 frames a second.
                "feeds": [{"path": herd["feeds"], "compute_device": stand.igpu, "counting": "herd"},
                          {"path": line["feeds"], "compute_device": other, "counting": "line"}],
            }),
            Wait(50.0),
        ),
        stop=("/api/smart-city-monitor/stop",),
        hold=0.0,
        at_most=130.0,
    )


# What the documents scene asks. Short enough to be read standing up, about
# the bundled folder, and checked on 2026-10-09 with the 1.5B model on the
# NPU: with the folder, both get the answer that is in the files; alone, the
# first is answered "the team's captain, on September 16, 2023" -- made up,
# word for word the same each time -- and the second "I don't have that
# information". The scene's story holds for either.
_DOCUMENT_QUESTIONS = (
    "Who makes the final go/no-go decision on the Lyon pilot, and on which date?",
    "What happened near Dock C during the Lyon workshop?",
)
_DOCUMENT_FOLDER = "meridian-rollout-2026"


def documents(stand: Stand, loop: int) -> Scene | Skip:
    """One question, asked twice of the same model: without the files, then
    with them. Asked for by the user on 2026-10-09, in place of a scene that
    answered a question beside a camera picture: "we can't really understand
    the principle behind the Q&A part"."""
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("The same question, without the files and with them", "La même question, sans les fichiers puis avec")
    if not (stand.npu or stand.igpu):
        return Skip(title, "needs an NPU or a GPU")
    sample = _sample(stand, "doc-qa", 0, lambda s: str(s.get("folder", "")).replace("\\", "/").rstrip("/").endswith(_DOCUMENT_FOLDER))
    if sample is None:
        return Skip(title, "its sample documents could not be found")
    files = [asset["name"] for asset in sample.get("assets") or [] if not asset.get("image")]
    question = _DOCUMENT_QUESTIONS[(loop - 1) % len(_DOCUMENT_QUESTIONS)]
    device = "NPU" if stand.npu else stand.igpu
    cancel = "/api/bricks/doc-qa/stop"
    return Scene(
        id="documents",
        title=title,
        demo="doc-qa",
        view="documents",
        beats=(
            Beat(say(
                "A question about a company's own files, the kind no model was trained on. First it is put to a small "
                "language model alone: nothing but the question is in the conversation.",
                "Une question sur les dossiers d'une entreprise, de ceux qu'aucun modèle n'a appris. Elle est d'abord "
                "posée à un petit modèle de langage seul : il n'y a que la question dans la conversation.",
            )),
            Beat(say(
                "That is the model alone. It has nothing to go on: it either says so or, worse, answers anyway with "
                "something it made up. The files on the left were never shown to it.",
                "Voilà le modèle seul. Il n'a rien sur quoi s'appuyer : soit il le dit, soit, pire, il répond quand "
                "même en inventant. Les fichiers à gauche ne lui ont jamais été montrés.",
            ), answer="alone"),
            Beat(say(
                "Now the folder is read, here on this laptop: each file is cut into passages, and each passage turned "
                "into numbers that say what it is about. " + ("All of it on the NPU, and the files go nowhere."
                                                              if stand.npu else "The files go nowhere."),
                "Le dossier est maintenant lu, ici, sur ce portable : chaque fichier est découpé en passages, et chaque "
                "passage traduit en nombres qui disent de quoi il parle. " + ("Tout cela sur le NPU, et les fichiers ne vont nulle part."
                                                                              if stand.npu else "Les fichiers ne vont nulle part."),
            ), answer="read"),
            Beat(say(
                "Same question, same model. This time the passages closest to it were put into the conversation, and "
                "the answer is written from them: the files it came from are marked on the left.",
                "Même question, même modèle. Cette fois, les passages les plus proches ont été glissés dans la "
                "conversation, et la réponse en est tirée : les fichiers dont elle vient sont marqués à gauche.",
            ), result=True),
        ),
        chips=(
            Chip(_NPU if stand.npu else _IGPU,
                 say("Reads the files and answers · two small models", "Lit les fichiers et répond · deux petits modèles"), "doc-qa"),
        ),
        props={"question": question, "files": files, "folder": _DOCUMENT_FOLDER},
        steps=(
            # Alone: no folder is indexed for it, only the models loaded.
            Ask("/api/doc-qa/ask", {"question": question, "alone": True, "engine": "openvino", "compute_device": device},
                cancel=cancel, keep=False, name="alone"),
            Wait(10.0),
            # Read again each time, a second's work: "now the folder is read"
            # is said of something that is happening.
            Ask("/api/doc-qa/ingest", {"folder": sample["folder"], "engine": "openvino", "compute_device": device, "reindex": True},
                keep=False, name="read"),
            Wait(10.0),
            Ask("/api/doc-qa/ask", {"question": question}, cancel=cancel, name="with"),
        ),
        # The two models are taken off their chip: left loaded, they sat
        # there through every other scene of the loop.
        stop=(cancel,),
        hold=22.0,
        at_most=200.0,
    )


def camera(stand: Stand, loop: int) -> Scene | Skip:
    """Whoever stands in front of the laptop, boxed by the detector and put
    into a sentence by the vision model. Kept at the user's wish on
    2026-10-09 ("it feels potentially engaging for people to be interacting
    with the demo camera"), with the commentary added to it.

    The sentence is plain, with no mood: the small model that gives a mood
    embroiders, and what it embroiders here would be about a visitor. And
    the vision model is asked what people do, never what they look like."""
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("The camera sees you", "La caméra vous voit")
    if not stand.igpu:
        return Skip(title, "needs a GPU for the vision model")
    if not stand.cameras:
        return Skip(title, "the camera was left out (tick “Use the camera”)" if stand.cameras_found else "needs a camera")
    watcher, watcher_chip = ("NPU", _NPU) if stand.npu else ("CPU", _CPU)
    return Scene(
        id="camera",
        title=title,
        demo="object-detection",
        view="camera",
        beats=(
            Beat(say(
                "That is this laptop's camera, and that is you. An object detector draws a box round everything it "
                "recognises, in every frame. Nothing is recorded, and the picture never leaves this machine.",
                "Voici la caméra de ce portable, et vous voici. Un détecteur d'objets encadre tout ce qu'il reconnaît, "
                "à chaque image. Rien n'est enregistré, et l'image ne quitte jamais cette machine.",
            )),
            Beat(say(
                "Move, or hold something up: a cup, a phone, a book. It knows eighty everyday things, and it runs on "
                + ("the NPU, one small model at a few watts." if stand.npu else "the processor: this machine has no NPU."),
                "Bougez, ou montrez un objet : une tasse, un téléphone, un livre. Il connaît quatre-vingts objets du "
                "quotidien, et tourne sur " + ("le NPU, un petit modèle à quelques watts." if stand.npu
                                               else "le processeur : cette machine n'a pas de NPU."),
            ), stage="default", after=9.0),
            Beat(say(
                "Meanwhile a much larger model, on the integrated GPU, is shown a frame every few seconds and says in "
                "a sentence what is going on. It is asked what people do, never what they look like.",
                "Pendant ce temps, un modèle bien plus gros, sur le GPU intégré, reçoit une image toutes les quelques "
                "secondes et dit en une phrase ce qui se passe. On lui demande ce que font les gens, jamais leur apparence.",
            ), stage="default", after=24.0),
            Beat(say(
                "Two models on two chips, watching the same camera: one fast and narrow, one slow and able to put it "
                "into words. Both run on this laptop, with no connection needed.",
                "Deux modèles sur deux puces, devant la même caméra : l'un rapide et étroit, l'autre lent mais capable "
                "de le dire avec des mots. Les deux tournent sur ce portable, sans aucune connexion.",
            ), stage="default", after=40.0),
        ),
        chips=(
            Chip(watcher_chip, say("Boxes what it sees · YOLO11s detector", "Encadre ce qu'il voit · détecteur YOLO11s"), "object-detection"),
            Chip(_IGPU, say("Says what is going on · Qwen2.5-VL 7B", "Dit ce qui se passe · Qwen2.5-VL 7B"), "video-commentary", ("vision",)),
        ),
        props={"chip": watcher_chip},
        steps=(
            Start("/api/object-detection/start", {
                "source": "webcam", "camera_index": stand.cameras[0], "engine": "openvino", "compute_device": watcher}),
            # One of the two opens the camera, and the other watches what it watches.
            Until("/api/object-detection/detections", "watching", True, timeout=60.0),
            Start("/api/video-commentary/start", {
                "source": "detector", "people": True, "mood": "plain", "voice": "",
                "vision_device": stand.igpu, "mood_device": "NPU" if stand.npu else stand.igpu,
            }),
            Wait(52.0),
        ),
        stop=("/api/video-commentary/stop", "/api/object-detection/stop"),
        hold=0.0,
        at_most=150.0,
    )


def video_commentary(stand: Stand, loop: int) -> Scene | Skip:
    """The experimental commentator: a video watched by a vision model, and
    its plain sentence said again in one voice after another."""
    say = lambda english, french: _say(stand, english, french)  # noqa: E731
    title = say("A video, watched and commented on", "Une vidéo regardée et commentée")
    if "video-commentary" in HELD_BACK:
        return Skip(title, HELD_BACK["video-commentary"])
    if not stand.igpu:
        return Skip(title, "needs a GPU for the vision model")
    sample = _sample(stand, "video-commentary", loop - 1, lambda s: s.get("ready") and s.get("path"))
    if sample is None:
        return Skip(title, "needs its sample videos fetched (Prepare models)")
    voice = "NPU" if stand.npu else stand.igpu
    watches = say("Watches the video · Qwen2.5-VL 7B", "Regarde la vidéo · Qwen2.5-VL 7B")
    speaks = say("Gives it a voice · Qwen2.5 1.5B", "Lui donne une voix · Qwen2.5 1.5B")
    if stand.npu:
        chips = (Chip(_IGPU, watches, "video-commentary", ("vision",)), Chip(_NPU, speaks, "video-commentary", ("mood",)))
    else:
        chips = (Chip(_IGPU, say("Watches, then gives a voice · two models", "Regarde, puis donne une voix · deux modèles"),
                      "video-commentary", ("vision", "mood")),)
    mood = lambda key: Start("/api/video-commentary/mood", {"mood": key})  # noqa: E731
    return Scene(
        id="video-commentary",
        title=title,
        demo="video-commentary",
        view="commentary",
        beats=(
            Beat(say(
                "A video plays from this laptop's disk, and a vision model watches it. Every few seconds it is shown "
                "one frame and asked a single question: what is happening here?",
                "Une vidéo est lue depuis le disque de ce portable, et un modèle de vision la regarde. Toutes les "
                "quelques secondes, on lui montre une image et on lui pose une seule question : que se passe-t-il ?",
            )),
            Beat(say(
                "That is its answer on the picture: one plain sentence, written on the integrated GPU in about a "
                "second. Nobody labelled this video; the model has never seen it before.",
                "Voici sa réponse, sur l'image : une phrase simple, écrite sur le GPU intégré en une seconde environ. "
                "Personne n'a annoté cette vidéo ; le modèle ne l'a jamais vue.",
            ), stage="vision", figure=True),
            Beat(say(
                "Now a second, small model says each sentence again in a voice: a sports commentator first, then a "
                "nature documentary. " + ("It runs on the NPU, so the vision model never waits for it."
                                          if stand.npu else "On this machine it shares the graphics chip."),
                "Un second modèle, tout petit, redit maintenant chaque phrase avec une voix : un commentateur sportif "
                "d'abord, puis un documentaire animalier. " + ("Il tourne sur le NPU : le modèle de vision ne l'attend jamais."
                                                               if stand.npu else "Sur cette machine, il partage la puce graphique."),
            ), stage="mood", figure=True),
            Beat(say(
                # Measured: a rider became "brave cowboys", a herd was "on its way to market".
                "The plain sentence stays under the picture, because the small model embroiders: what it adds is "
                "style, not something it saw. The comments are in English, whatever language this story is told in.",
                "La phrase simple reste sous l'image, car le petit modèle brode : ce qu'il ajoute est du style, pas "
                "quelque chose qu'il a vu. Les commentaires sont en anglais, quelle que soit la langue de ce récit.",
            ), stage="mood", figure=True),
        ),
        chips=chips,
        props={
            "video": sample["name"],
            "moods": {
                "plain": say("Just what it sees", "Ce qu'il voit, sans plus"),
                "sports": say("Sports commentator", "Commentateur sportif"),
                "documentary": say("Nature documentary", "Documentaire animalier"),
                "upbeat": say("Upbeat", "Enjoué"),
            },
        },
        steps=(
            Start("/api/video-commentary/start", {
                # No sound: a stand at a large event is too loud for it.
                "source": "file", "path": sample["path"], "loop": True, "mood": "plain", "voice": "",
                "vision_device": stand.igpu, "mood_device": voice,
            }),
            # The two models load for a quarter of a minute, more on a cold
            # machine: the voices wait for the first plain sentence, not for a clock.
            Until("/api/video-commentary/comments", "commenting", True, timeout=120.0),
            Wait(12.0),
            mood("sports"),
            Wait(20.0),
            mood("documentary"),
            Wait(20.0),
            mood("upbeat"),
            Wait(12.0),
        ),
        stop=("/api/video-commentary/stop",),
        hold=0.0,
        at_most=210.0,
    )


# In the order they play, each under the key the start screen ticks it by.
# Heavy and light alternate: the 30B model lost a fifth of its speed after
# many runs in a row (docs/AUTO_DEMO.md).
#
# The two counting scenes share one place and take turns in it (asked for by
# the user on 2026-10-09): the streets on the first turn of the loop, the
# herd and the line on the second, never both in one turn. If one of them
# cannot play here, or was not chosen, the other plays every turn.
#
# The camera comes early: it is the scene a passer-by can play with. The two
# scenes that load the 7B vision model (the camera, the commentator) are
# kept apart, and so are the two that put boxes on a picture.
PLAYLIST: list[Slot] = [
    Entry("page-agent", page_agent),
    Entry("camera", camera),
    Entry("expense-extraction", expense_extraction),
    (Entry("smart-city", smart_city), Entry("herd-and-line", herd_and_line)),
    Entry("documents", documents),
    Entry("video-commentary", video_commentary),
]
