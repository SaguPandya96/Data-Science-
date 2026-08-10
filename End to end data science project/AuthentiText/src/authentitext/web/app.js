const input = document.querySelector("#text-input");
const count = document.querySelector("#character-count");
const analyzeButton = document.querySelector("#analyze-button");
const clearButton = document.querySelector("#clear-button");
const analyzeAnotherButton = document.querySelector("#analyze-another-button");
const formError = document.querySelector("#form-error");
const guidePanel = document.querySelector("#guide-panel");
const resultPanel = document.querySelector("#result-panel");

const categoryCopy = {
  likely_human: { label: "Likely human-written", icon: "H" },
  uncertain: { label: "Uncertain", icon: "?" },
  likely_machine: { label: "Likely machine-generated", icon: "M" },
};

function wordCount(value) {
  const trimmed = value.trim();
  return trimmed ? trimmed.split(/\s+/u).length : 0;
}

function updateCount() {
  const characters = input.value.length;
  const words = wordCount(input.value);
  count.textContent = `${characters.toLocaleString()} characters · ${words.toLocaleString()} words`;
}

function setLoading(loading) {
  analyzeButton.disabled = loading;
  analyzeButton.classList.toggle("loading", loading);
  analyzeButton.querySelector(".button-label").textContent = loading ? "Analyzing…" : "Analyze text";
  input.setAttribute("aria-busy", String(loading));
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function clearError() {
  formError.textContent = "";
  formError.hidden = true;
}

function renderList(element, items, messageSelector = (item) => item.message) {
  element.replaceChildren();
  items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = messageSelector(item);
    element.append(listItem);
  });
}

function renderResult(result) {
  const copy = categoryCopy[result.category] ?? categoryCopy.uncertain;
  const likelihood = Math.max(0, Math.min(100, result.calibrated_machine_likelihood * 100));
  const lowerThreshold = Math.max(
    0,
    Math.min(100, result.thresholds.likely_human_max * 100),
  );
  const upperThreshold = Math.max(
    0,
    Math.min(100, result.thresholds.likely_machine_min * 100),
  );
  resultPanel.dataset.category = result.category;
  document.querySelector("#result-category").textContent = copy.label;
  document.querySelector("#result-icon").textContent = copy.icon;
  document.querySelector("#likelihood-value").textContent = `${likelihood.toFixed(1)}%`;
  document.querySelector("#likelihood-fill").style.width = `${likelihood}%`;
  document.querySelector("#likelihood-marker").style.left = `${likelihood}%`;
  document.querySelector("#lower-threshold").style.left = `${lowerThreshold}%`;
  document.querySelector("#upper-threshold").style.left = `${upperThreshold}%`;
  const meter = document.querySelector("#likelihood-meter");
  meter.setAttribute("aria-valuenow", likelihood.toFixed(1));
  meter.setAttribute("aria-valuetext", `${likelihood.toFixed(1)} percent machine likelihood`);

  const badge = document.querySelector("#evidence-badge");
  badge.textContent = result.evidence_quality === "low" ? "Low evidence" : "Standard evidence";
  badge.classList.toggle("low", result.evidence_quality === "low");

  const warningSection = document.querySelector("#warning-section");
  warningSection.hidden = result.warnings.length === 0;
  renderList(document.querySelector("#warning-list"), result.warnings);
  renderList(document.querySelector("#limitation-list"), result.limitations);

  const tokens = result.input_summary.whitespace_tokens.toLocaleString();
  const characters = result.input_summary.characters.toLocaleString();
  document.querySelector("#result-size").textContent = `${tokens} tokens · ${characters} characters`;
  document.querySelector("#model-id").textContent =
    `Model ${result.model.base_model_sha256.slice(0, 12)}… · calibration ` +
    `${result.model.calibration_sha256.slice(0, 12)}…`;

  guidePanel.hidden = true;
  resultPanel.hidden = false;
  resultPanel.focus({ preventScroll: true });
}

async function analyze() {
  clearError();
  if (!input.value.trim()) {
    showError("Paste a passage before analyzing.");
    input.focus();
    return;
  }
  setLoading(true);
  try {
    const response = await fetch("/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: input.value }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.error?.message ?? "The analysis could not be completed.");
    }
    renderResult(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "The analysis could not be completed.";
    showError(message);
  } finally {
    setLoading(false);
  }
}

function reset() {
  input.value = "";
  updateCount();
  clearError();
  resultPanel.hidden = true;
  guidePanel.hidden = false;
  input.focus();
}

input.addEventListener("input", updateCount);
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    analyze();
  }
});
analyzeButton.addEventListener("click", analyze);
clearButton.addEventListener("click", reset);
analyzeAnotherButton.addEventListener("click", reset);
input.value = "";
updateCount();
