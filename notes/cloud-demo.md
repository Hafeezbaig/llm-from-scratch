# Trying the demo on Google Cloud, explained simply (Cloud Run)

## The idea as a story
- **Training was baking the cake.** It needed a hot GPU kitchen, and it's done: the brain is saved in `checkpoints/gpt.pt` (45 MB).
- **Stage 7 lets people taste it.** A small web page: type a start like "Holmes said", press a button, get text back.
- **Cloud Run is a food stall that opens only when a customer walks up.** When nobody is there, it closes by itself, and you pay nothing while it's closed.
- **This is only a test.** We put the stall up, try it, then **take it down completely**. To try it again later, ask the AI agent: "deploy the Stage 7 demo on GCP for a test". It takes about 5–10 minutes to put back up.

## Why no GPU is needed
Writing text is far lighter than training. Measured on the Mac's CPU alone (2026-11-17):

| CPU threads | 300 characters took |
|---|---|
| 1 | 1.6 s |
| 2 | 1.7 s (no faster: the sums are too small to share out) |

A cloud CPU is a bit slower than the Mac's, so expect a few seconds. **One CPU is enough.**

## The pieces (built in Stage 7)
| File | Its job |
|---|---|
| `app.py` | the web server (FastAPI). It loads the brain **once** when it starts, then answers requests. `/generate` takes a start text and the Stage 5 knobs and returns the text; `/` shows the web page. |
| the web page | a text box, the knobs, and a "Write" button |
| `Dockerfile` | the packing list: Python, PyTorch (the CPU-only version, much smaller), FastAPI, the code and the brain file |
| `.gcloudignore` | what **not** to upload. Without it, Google reuses `.gitignore`, which skips `*.pt`, so the brain would be left behind. It also keeps `internal-docs/` (private), `.venv/` and `data/` off the cloud. |

## What happens when it runs
```
you: "deploy the Stage 7 demo on GCP for a test"
 1. upload: the project folder (minus .gcloudignore) goes to a Google storage bucket
 2. pack:   Cloud Build follows the Dockerfile and packs everything into one sealed box (a "container image")
 3. shelve: the box is stored in Artifact Registry
 4. serve:  Cloud Run opens the box when someone visits, and gives a link like https://llm-demo-xxxx.run.app
 5. test:   open the link, type a start, get text
 6. delete: the service, the box and the upload bucket, so nothing is left
```

## How it differs from training on the cloud
| | Training (SkyPilot, `notes/cloud-gpu.md`) | The demo (Cloud Run) |
|---|---|---|
| What you rent | a whole GPU machine | a slice of a CPU, only while answering |
| Billed | every second it's on | only while a request is being answered |
| When nobody uses it | it switches off after 10 idle minutes | it switches off by itself, almost straight away |
| Ends with | `cloud down` | deleting the service, the box and the bucket |

## The settings and why
| Setting | Value | Why |
|---|---|---|
| region | `us-central1` (Iowa) | one of the cheapest regions; the distance adds well under a second |
| CPU / memory | 1 CPU, 2 GiB | the speed test above; PyTorch and the 45 MB brain fit comfortably |
| fewest machines | 0 | closes when nobody is using it, so there's no cost while idle |
| most machines | **1** | a flood of visitors can't open more stalls and run up a bill |
| who can open it | anyone with the link | so you can try it from your phone too. It's deleted after the test. |
| longest answer | 1,000 characters (set in `app.py`) | one request can't keep the CPU busy for minutes |

## The cost
- **Cloud Run:** you pay only while it's answering, and Google gives a free monthly allowance. A test of an hour or so should be free.
- **Packing the box:** a few minutes of Cloud Build per deploy, which is a small cost at most.
- **Storing the box:** a few cents a month, but we delete it after the test.
- **Expected total for one test: $0 to a few cents.**
- Turning a Google service on (enabling its API) is free, so they can stay on.

## The slow first visit
Because the stall closes when idle, the first visitor waits while it opens: Python starts, PyTorch loads, then the brain loads. That's called a **cold start**. My guess is 10–20 seconds; the real number gets added here after the first test. Later visits are quick.

## Runbook (for the AI agent)
**Rules:**
- Ask the user before deploying: it makes a public link.
- Always deploy with `--max-instances 1`.
- `.gcloudignore` must exist before deploying, and it must keep `internal-docs/` out (private) and `checkpoints/gpt.pt` in.
- When the user says they're done, delete everything (step 3) and check (step 4). If the session ends before that, remind the user that it's still up.
- Never put keys in the repo. The Google login lives in `~/.config/gcloud`.

**Before the first deploy (2026-11-17)** the project had no Cloud Run services, no Artifact Registry repositories and no storage buckets. So anything of those kinds that appears was made by the deploy and is safe to delete.

```bash
# 0. Once per project (free): turn on the three services
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 1. Deploy from the project root (about 5–10 minutes)
gcloud run deploy llm-demo --source . --region us-central1 \
  --cpu 1 --memory 2Gi --min-instances 0 --max-instances 1 \
  --allow-unauthenticated --quiet

# 2. Test (or just open the link in a browser)
URL=$(gcloud run services describe llm-demo --region us-central1 --format='value(status.url)')
curl -s -X POST "$URL/generate" -H 'Content-Type: application/json' \
  -d '{"prompt": "Holmes said", "tokens": 200, "temperature": 0.8}'

# 3. Delete everything
gcloud run services delete llm-demo --region us-central1 --quiet
gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1 --quiet
gcloud storage ls        # the deploy's upload bucket has "run-sources" or "cloudbuild" in its name
gcloud storage rm -r gs://<that-bucket>

# 4. Check that nothing is left (all three should be empty)
gcloud run services list
gcloud artifacts repositories list
gcloud storage ls
```

## Words you'll meet
- **Container image:** a sealed lunchbox holding the code, Python, PyTorch and the brain. It runs the same anywhere.
- **Dockerfile:** the recipe for packing the lunchbox.
- **Cloud Build:** Google's packing service.
- **Artifact Registry:** Google's shelf for packed boxes.
- **Cloud Run:** opens the box when someone visits, and closes it when nobody does.
- **Endpoint:** an address on the server that does one job, like `/generate`.
- **Cold start:** the wait while a closed stall opens.
