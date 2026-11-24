# X thread

**1/**
Presenting "Persona-Model Collapse in Emergent Misalignment" at #NeurIPS2026 in Sydney next month! [tag UChicago DSI]

Fine-tuning LLMs on narrow harmful data makes them broadly misaligned. We propose it also erodes how models represent and distinguish personas. 🧵

> Figure: `heading-figure.png`

---

**2/**
Role-playing diverse personas on questionnaires, misaligned models give more polarized, all-or-nothing answers.

GPT-4o: endpoint answers rise from 19% to 73% (48% under a benign fine-tuning control).

> Figure: `response_distribution_mfq.png`

---

**3/**
Answers also become less stable: repeating the same question under the same persona yields more inconsistent responses, beyond what benign fine-tuning causes, across four model families.

> Figure: `bar_robustness.png`

---

**4/**
Inside Qwen3-235B, persona representations span fewer effective dimensions in the misaligned model than in the original and benign control, most clearly in the final layers.

> Figure: `qwen235b_pr_panels.png`

---

**5/**
Joint work with Renato Vicente.

📄 https://arxiv.org/abs/2605.12850

Come chat at our poster: Tue, Dec 8, 2026 • 5:00 PM – 8:00 PM AEDT.
