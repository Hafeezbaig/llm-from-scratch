# Training on a cloud GPU, explained simply (SkyPilot on AWS and GCP)

## The idea as a story
- **Your Mac is your home kitchen.** Training a model is like baking 100 cakes, so the kitchen gets hot.
- **AWS and GCP are two companies that rent out big kitchens by the minute.** Their kitchens have GPUs, which are special ovens built for this kind of work.
- **SkyPilot is a helper.** It books a kitchen, carries your ingredients (your code) there, does the baking, and brings the cakes back (the trained model and the graph). Then it hands back the keys, so the meter stops.

SkyPilot is free and open source. The machines run in **your own** cloud account, and your keys stay on your Mac.

## What happens when it runs
```
you: "train this on the cloud"
 1. SkyPilot rents a GPU machine in YOUR cloud account
 2. copies your project folder to it
 3. installs what's in requirements.txt
 4. runs python train.py
 5. the results are copied back to your Mac
 6. the machine switches itself off after 10 idle minutes
```

All it needs is a tiny recipe file in the project:
```yaml
resources:
  accelerators: L4:1        # which GPU, and how many
workdir: .                  # copy this project folder up
setup: pip install -r requirements.txt
run: python train.py
```

These are the commands (Claude can type them for you):

| Command | What it does |
|---|---|
| `sky check` | checks that your AWS/GCP login works |
| `sky show-gpus` | lists GPUs and today's prices |
| `sky launch -c llm task.yaml -i 10` | rents a machine, runs the job, auto-stops after 10 idle minutes |
| `sky exec llm task.yaml` | runs again on the same machine (fast, no new setup) |
| `sky status` | shows what's running and costing money |
| `sky stop llm` / `sky down llm` | park the machine / give it back completely |

**Stop vs down:**
- **`stop`** is like parking a rented car. You don't pay for driving, but you still pay a small fee for the disk.
- **`down`** is like returning the car. You pay nothing, but the next start has to set everything up again.

## AWS vs GCP, side by side
| | **GCP (Google)** | **AWS (Amazon)** |
|---|---|---|
| Keep it separate | Make a new **project**, and delete it later to remove everything | Make a separate **IAM user** with limited permissions |
| Log in once | `gcloud auth login` and `gcloud auth application-default login` | `aws configure` (access key) |
| GPU permission (starts at 0) | Ask for the **"GPUs (all regions)"** quota → 1 | Ask for **"Running On-Demand G and VT instances"** → 8 (counted in CPUs, which is confusing) |
| Cheap GPU (T4) | about $0.55/hour | about $0.55/hour (`g4dn.xlarge`) |
| Faster GPU (L4) | about $0.75/hour (`g2-standard-4`) | about $0.80/hour (`g6.xlarge`) |
| Billing | per second | per second |
| Spending alert | Billing → Budgets & alerts | AWS Budgets |

These prices are rough (late 2026), and `sky show-gpus` shows today's real ones. A 20-minute training run costs well under $1.

## Words you'll meet
- **VM or instance:** a rented computer.
- **GPU:** a chip that does thousands of small sums at once. Training is mostly sums.
- **T4 and L4:** GPU models. The T4 is old and cheap, and for a small model like ours it's only about as fast as a Mac; the gain is that your Mac stays cool. The L4 is newer and clearly faster.
- **Quota:** a limit on how much you're allowed to rent. New accounts get 0 GPUs, so you ask once. Approval can take from a few hours to a couple of days.
- **Region:** which city the data centre is in. SkyPilot picks the cheapest one that has GPUs free.
- **Spot:** spare machines sold about 60% cheaper, but the company can take them back at any moment. Fine for short runs.

## Staying safe with money
1. **Auto-stop (`-i 10`)** is the real safety net. A forgotten machine switches itself off.
2. **A budget alert** only *emails* you. It doesn't stop spending by itself, which surprises most people.
3. **`sky status`** shows anything still running. Check it at the end of every session.
4. **A separate project or user** keeps this work away from anything else in your account.

## Which one to pick
**GCP** is slightly easier for one person working alone. Projects keep things neatly separate, and its GPU permission is counted in GPUs, not CPUs. If you already have credits or bills on AWS, use AWS instead; SkyPilot works the same on both, and can even use both and pick the cheaper one.

## The one-time setup
- **You (about 10 minutes):**
  - Make a separate project (GCP) or user (AWS), set a $20 budget alert, and log in once.
  - Ask for 1 GPU of quota, then wait for approval.
- **Claude:**
  - Install SkyPilot and run `sky check`.
  - Do one small test run.
  - Save a short instruction file so any future Claude session knows to send heavy jobs to the cloud.
