# Gemini as a development reviewer

Use Gemini to review actual source, propose experiments and critique development images. The project's planner is separately owned by the mission/integration workstream. Do not add a competing planner or change the run schema merely to spend credits.

## Tools

- Gemini Code Assist in VS Code: in-editor code explanation and repository review. Start with a read-only review, then inspect any proposed source diffs.
- Gemini CLI: a terminal coding agent, including API-key authentication. This is an alternative to the VS Code extension, not an additional prerequisite.
- Google AI Studio: browser-based prompt and image-review experiments using models available to your account.
- Google GenAI Python SDK (google-genai): a later option for repeatable API-based QA reports. This is not required to generate or validate scenes.

Credits are tied to their API/backend/project. A Google sign-in for Code Assist does not by itself prove that a particular API credit grant is being used. Confirm the credited project in AI Studio or the relevant Google Cloud console. Keep API keys out of code, screenshots and repository settings files. Use a user-level environment variable or the tool's supported credential interface.

## Repository review prompt

Read PLAN.md, sim/scene.py, tools/validate_run.py, tools/preview_scene.py, tools/check_scene.py, docs/scene-data.md and docs/scene-quality.md. Audit only the scene/data workstream. First return findings without editing files. For each finding, give the file, relevant line/function, impact, and a concrete fix. Check the JSON contract, seed reproducibility, same-layout comparisons across scenarios, burial/z interpretation, object IDs, separation of ground truth from agent input, and whether tests actually cover each claim. Distinguish locally verified facts from things requiring Isaac/OceanSim. Do not change agent/, perception/, modal/, run_demo.py, tests/test_smoke.py, shared PLAN.md, or output schemas. Do not invent a sonar annotation API. Do not run GPU jobs, commit, push, or publish.

## Preview review prompt

This image is a ground-truth development layout of four simulated seabed objects. It is not sonar output or a robot classification result. Review readability, hierarchy, contrast, labels, axes, legend and whether anything could misrepresent simulated evidence. Give up to five actionable findings. Do not infer real UXO identification, acoustic realism, detection accuracy or operational safety from this map.

## Useful later multimodal QA

Once real rendered frames exist, Gemini can propose annotation issues from a sonar frame plus its bounding-box overlay. Treat those as review suggestions, not authoritative labels. Validate against actual simulator geometry/visibility and human inspection. Structured JSON responses can make a repeatable issue list, but schema compliance does not establish correct sonar interpretation. Save optional reviews outside required run contracts, for example under reports/.

Official references:

- https://docs.cloud.google.com/gemini/docs/codeassist/use-agentic-chat-pair-programmer
- https://geminicli.com/docs/get-started/installation/
- https://geminicli.com/docs/get-started/authentication/
- https://ai.google.dev/gemini-api/docs/structured-output
- https://ai.google.dev/gemini-api/docs/image-understanding

Documentation checked 7 November 2026. Verify currently available models and account permissions in your own environment rather than assuming credits remove rate limits or compatibility constraints.
