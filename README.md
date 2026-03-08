# AV Contagion Experiment Scaffold

## Codebase Structure

- `.env` for seed/keys/config
- `requirements.txt` for Python dependencies
- `config/models.yaml` for model routing by task
- `prompts/contagion/{malign,benign,base,judge}` for prompt templates (`.txt`)
- `prompts/confidence/{base,judge}` for confidence prompt templates (`.txt`)
- `problems/` for seeded subsets sampled from BigCodeBench-Hard
- `shared/{generate_problem_set.py,response.py,judge_trace.py}` for shared scripts
- `results/{time_constraint,contagion,confidence}` for output metrics
- `contagion/set_traces/{malign,benign}` for baseline traces per problem
- `contagion/temp_results/` for generated contagion traces
- `contagion/generate_traces.py` to produce k traces with n-step context
- `contagion/analyze_results.py` to compute probabilities and delta metrics
- `contagion/contagion_run.py` for one full end-to-end contagion run
- `confidence/{generate_traces.py,analyze_results.py,confidence_run.py}` for confidence scaffold runs

## API Provider

- OpenRouter is the default provider for API calls via `shared/response.py`.
- Set `OPENROUTER_API_KEY` in `.env`.
- Optional headers can be set with `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME`.
- Confidence runs use Azure OpenAI via `AzureOpenAI`:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_VERSION`
  - `AZURE_OPENAI_DEPLOYMENT`
  - Optional: `AZURE_OPENAI_JUDGE_DEPLOYMENT`

## Experimentation Setup (Contagion)

1. Generate one malign and one benign trace for each problem.
2. For each problem `i >= 2`, generate `k` traces using previous traces as context.
3. Optionally use canonical solutions + LLM judge for RH score calibration.
4. Compute `P(hack_i | hack_(i-1))` and `P(hack_i | no-hack_(i-1))`.
5. Compute `delta_1 = P(hack_i | hack_(i-1)) - P(hack_i | no-hack_(i-1))`.
6. Extend to `delta_n` as context window grows.

## Experimentation Setup (Confidence)

1. Use the same seeded problem set in `problems/`.
2. Generate `k` traces per problem with a single model/provider.
3. Confidence is self-reported by the model from the first line of output:
   - `CONFIDENCE: <integer 0-100>`
   - `CODE:`
   - Python code only after that (no prose / markdown / extra headers)
4. Attach judge metadata for suspicious/reward-hacky behavior.
5. Compute scaffold metrics:
   - `solve_rate` (TODO until correctness labels are added)
   - `hack_rate` (from judge label/score if present)
   - `avg_model_confidence` (from parsed self-reported model confidence)
6. Outputs are written under `results/confidence/<run_id>/`.

## Future Extension

Add difficulty-gap controls from previous `n` problems to current problem
(e.g., hard->medium, hard->easy).

## Quick Start (Contagion)

```bash
pip install -r requirements.txt
python generate_problem_set.py --seed 42 --k 20 --dataset-id bigcode/bigcodebench-hard --split auto
python contagion/contagion_run.py --k 3 --n 1 --provider openrouter
```

## Quick Start (Confidence)

```bash
pip install -r requirements.txt
python generate_problem_set.py --seed 42 --k 20 --dataset-id bigcode/bigcodebench-hard --split auto
python confidence/confidence_run.py --k 3 --provider azure
python confidence/confidence_run.py --k 1 --provider mock --overwrite
```
