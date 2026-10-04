# Email in your own voice: plan for the drafting demo

Status: plan, not built, and set aside on 2026-10-04: the user is not
convinced by this demo. Kept for the measurements. It covers the "Inbox Triage &
Draft Assistant" card (`smart-inbox`), with the aim the user set for it: the
laptop learns how a person writes from the mail they have sent, then drafts
new mail the way they would.

## What the audience sees

1. **Learn my voice.** Point the demo at a folder of sent emails. The NPU
   reads and indexes them; the GPU writes a *style card*: a dozen lines on
   how this person greets, opens, lays out actions, closes and signs, quoting
   their recurring phrases. The card is on screen and can be edited.
2. **Answer an email.** Pick an incoming email and type, in a few words, what
   the reply should say.
3. **Two drafts side by side**, written as you watch: the generic one any
   assistant gives, and one in your voice, built from the style card and the
   three emails you wrote that are most like this situation.
4. **A check under the draft**: every date, amount and name in it that is in
   neither the email being answered nor your notes is marked.
5. Nothing is sent. Copy the draft, or save it as an `.eml`.

The story is the one only a local machine can tell: the most private text a
person has is read, learned from and written with, and none of it leaves the
laptop. And it uses two chips for two jobs: the NPU reads, the GPU writes.

## What was measured

A feasibility test on the XPS 14, with a fictional writer who has a marked
style (short sentences, an opening "Short version:" or "Quick one:", actions
as `1) owner -> task -> date`, a fixed closing line, no stock phrases), eight
of their sent emails, and one formal email to answer. Each draft was scored
against seven habits the writer has; their own emails average 5.8 of 7 (not
every email has a list).

| Model and chip | No guidance | Style card | Card and 3 examples | 3 examples only | One draft |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Coder-30B, integrated GPU | 3 of 7 | 6 of 7 | 6 of 7 | 6 of 7 | 3.3 s at 46 tok/s |
| Qwen2.5-1.5B, NPU | 1 of 7 | 1 of 7 | 2 of 7 | 1 of 7 | 1.5-3.5 s at 59 tok/s |

- **The large model picks up a voice from very little.** Unguided it wrote
  the usual email (12.5 words a sentence, "Best regards"); guided it wrote
  8 to 10 words a sentence against the writer's 7.5, with their greeting,
  list layout, closing line and signature. In the draft that was read in
  full, the one habit scored as missed was a blank line the scoring was too
  strict about.
- **It learns fast.** The style card took 3.6 s to write from eight emails;
  the model loads in 16 s.
- **The small model cannot write in a voice.** Its style card invented a
  habit, and one draft answered as the wrong person. It stays useful for
  sorting and summarising, not for writing.
- **Guided drafts invent content.** The best-styled draft made up a deadline
  ("Mon 24 Sep") and a list of who does what that were in nobody's notes.
  Style transfers; facts have to be held down separately.

One writer, one email, one run per cell, and a habit list written to fit
that writer: enough to choose an approach, not a quality claim.

## How it learns the way we write

| | What it is | For | Against |
| --- | --- | --- | --- |
| **A. Style card** | The model reads the sent mail and writes down the habits; the card goes into every drafting prompt | Cheap (130 tokens), readable, editable: the audience sees what was learned and the user can correct it | Keeps surface habits, loses the feel of whole emails |
| **B. Their own emails as examples** | An index of the sent mail; for each reply, the three past emails closest to the situation go into the prompt | Carries real wording, length and layout; the index already exists in document Q&A | 300-600 prompt tokens; old facts can leak into the new email |
| **C. Fine-tuning** | Train a small adapter (LoRA) on the sent mail, load it into the model | The strongest claim: "it trained on my mail, on this laptop" | Not possible today without new parts: the installed PyTorch is CPU-only and there is no training library; needs a model small enough to train here, a conversion step, minutes to hours, and a way to show it did anything. OpenVINO GenAI can load adapters, which is the easy half |

**Proposed: A and B together.** The card makes the learning visible and
correctable; the examples carry the wording. C is a later experiment, worth
two days to find out whether it is feasible, and only if a show needs the
word "training".

"We" can also be a team: the same two steps over a shared folder give one
card for the team's voice, or one per sender.

## Holding the facts down

The rule for the user is *you say what, it says how*.

- The draft may use only what is in the email being answered and in the
  notes typed for this reply. The prompt says so, and says to leave a gap
  rather than fill one.
- After writing, a plain check (no model) pulls every date, number, amount
  and capitalised name out of the draft and marks those found in neither
  source. Same stance as the expense review: flag, don't trust.
- Example emails are stripped of what must not travel: the check above
  catches a date carried over from an old email.
- The demo never sends anything and holds no mail account.

## Where the emails come from

- **On stage: a bundled fictional mailbox** in `sample-data/`, about forty
  sent emails by one invented writer of the Meridian sample company, and a
  handful of incoming ones. Real mail does not belong on a projector.
- **For a one-to-one demo: your own folder.** First version reads a folder
  of `.eml` and `.txt` files (what Thunderbird, the new Outlook and most
  exports produce); `.mbox` next; classic Outlook `.msg` later, as it needs
  a parser library. No live mailbox connection: no credentials in the app,
  and it must work offline.
- **Only what you wrote counts.** Keep messages from the chosen sender,
  cut quoted replies, signatures and legal footers, skip automatic replies.
- **It can forget.** One button deletes the index and the style card.
  Nothing from the mail goes into the activity log beyond counts.
- A person who writes in two languages needs a card for each, and examples
  in the language of the email being answered.

## The brick

| Piece | Reused | New |
| --- | --- | --- |
| Reading mail | -- | parse `.eml`, keep the sender's own text |
| Index and "most similar emails" | document Q&A's embeddings and search, on the NPU | one index per mailbox |
| Style card | the large model through `create_llm`, streamed | the prompt; the card stored as editable text |
| Drafts | the same model; text as it is written, Stop, "Same page every time" | prompt assembly; two drafts side by side |
| Fact check | the expense review's stance | dates, numbers, names against the sources |
| Sorting the inbox (later) | the small model on the NPU | a summary and a priority per email |

**Which model writes.** The 30B coding model is already on the machine and
did the job. A general 30B model of the same family (the one R26 names) may
write better prose, at the cost of another 17 GB to download and keep.

## Steps

| Step | Delivers | Size |
| --- | --- | --- |
| **1** | The stage demo: fictional mailbox, style card, two drafts side by side, fact check | 3-4 days |
| **2** | Your own folder: `.eml` and `.mbox`, sender filter, quote stripping, forget button, a card per language | 2-3 days |
| **3** | Sorting on the NPU: a summary and a priority for each incoming email | about 2 days |
| **4** | Experiment: fine-tuning on the device | 2 days to find out |

Step 1 alone is the demo. It needs no new model and no new dependency.

## How we will know it works

- For the bundled writer, the habit score above: guided drafts at or near
  the writer's own, unguided ones well below.
- The fact check marks nothing the user did not write, on the bundled inbox.
- Side by side on screen, someone who has read three of the writer's emails
  picks the voiced draft without being told which it is.
- Style card in under 30 s for forty emails; a draft in under 5 s.

## To decide

1. **Whose voice on stage**: the bundled fictional writer (proposed), or
   real mail?
2. **One person or a team voice** first?
3. **The writing model**: reuse the 30B coder (no download), or add the
   general 30B?
4. **Is sorting the inbox in the first version**, or is step 1 drafting only?
5. **Is the fine-tuning experiment wanted**, for the "it trained here" claim?
