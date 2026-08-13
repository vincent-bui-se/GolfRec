const state = {
  result: null,
  activeTab: "drivers",
};

const form = document.querySelector("#profile-form");
const emptyState = document.querySelector("#empty-state");
const recommendations = document.querySelector("#recommendations");
const tabs = document.querySelector("#tabs");
const driverBudget = document.querySelector("#driver-budget");
const ironBudget = document.querySelector("#iron-budget");
const driverCondition = document.querySelector("#driver-condition");
const ironCondition = document.querySelector("#iron-condition");
const loadingState = document.querySelector("#loading-state");
const formError = document.querySelector("#form-error");
const formErrorDetail = document.querySelector("#form-error-detail");
const retryAction = document.querySelector("#retry-action");
const resultStatus = document.querySelector("#result-status");
const USED_MAX_YEAR = new Date().getFullYear() - 4;

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formValue(name) {
  const field = form.elements[name];
  if (!field) {
    return "";
  }
  if (field instanceof RadioNodeList) {
    return field.value;
  }
  return field.value;
}

function collectPayload() {
  return {
    shopping_for: formValue("shopping_for"),
    score_mode: formValue("score_mode"),
    handicap: Number(formValue("handicap")),
    average_score: Number(formValue("average_score")),
    speed_mode: formValue("speed_mode"),
    swing_speed: Number(formValue("swing_speed")),
    driver_carry: Number(formValue("driver_carry")),
    driver_shot_shape: formValue("driver_shot_shape"),
    driver_trajectory: formValue("driver_trajectory"),
    driver_goal: formValue("driver_goal"),
    iron_shot_shape: formValue("iron_shot_shape"),
    iron_goal: formValue("iron_goal"),
    iron_trajectory: formValue("iron_trajectory"),
    iron_feel: formValue("iron_feel"),
    iron_miss: formValue("iron_miss"),
  };
}

function titleCaseSpec(value) {
  return String(value || "-")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function updateSpecGrid(specs, result) {
  const tiles = document.querySelectorAll(".spec-tile strong");
  tiles[0].textContent = result.wants_driver ? `${specs.driver_loft} deg` : "-";
  tiles[1].textContent = result.wants_driver ? specs.shaft_flex : "-";
  tiles[2].textContent = result.wants_irons
    ? titleCaseSpec(specs.iron_category)
    : "-";
}

function updateConditionalFields() {
  const scoreMode = formValue("score_mode");
  document.querySelectorAll(".score-field").forEach((field) => {
    field.classList.toggle("hidden", field.dataset.scoreMode !== scoreMode);
  });

  const speedMode = formValue("speed_mode");
  document.querySelectorAll(".speed-field").forEach((field) => {
    field.classList.toggle("hidden", field.dataset.speedMode !== speedMode);
  });

  const shoppingFor = formValue("shopping_for");
  document
    .querySelectorAll(".driver-fields")
    .forEach((section) =>
      section.classList.toggle("hidden", shoppingFor === "Irons"),
    );
  document
    .querySelectorAll(".iron-fields")
    .forEach((section) =>
      section.classList.toggle("hidden", shoppingFor === "Driver"),
    );
}

function selectForDisplay(clubs, defaultLimit = 5) {
  if (clubs.length <= defaultLimit) {
    return clubs;
  }
  const firstFive = clubs.slice(0, defaultLimit);
  const firstScore = firstFive[0].score;
  if (firstFive.some((club) => club.score !== firstScore)) {
    return firstFive;
  }
  const selected = [...firstFive];
  for (const club of clubs.slice(defaultLimit)) {
    selected.push(club);
    if (club.score < firstScore) {
      break;
    }
  }
  return selected;
}

function filterByCondition(clubs, condition) {
  if (condition !== "used") {
    return clubs;
  }
  return clubs.filter(
    (club) => Number.isInteger(club.year) && club.year <= USED_MAX_YEAR,
  );
}

function filterByBudget(clubs, budget) {
  return clubs
    .filter((club) => typeof club.msrp === "number" && club.msrp <= budget)
    .sort((a, b) => b.score - a.score);
}

function maxMsrp(clubs, fallback) {
  const prices = clubs
    .map((club) => club.msrp)
    .filter((price) => typeof price === "number" && Number.isFinite(price));
  if (!prices.length) {
    return fallback;
  }
  return Math.ceil(Math.max(...prices) / 25) * 25;
}

function setBudgetStart(slider, clubs, fallback) {
  const highestPrice = maxMsrp(clubs, fallback);
  slider.max = String(highestPrice);
  slider.value = String(highestPrice);
}

function updateBudgetForCondition(slider, clubs, condition, fallback) {
  setBudgetStart(slider, filterByCondition(clubs, condition), fallback);
}

function renderClubList(listElement, countElement, clubs, budget, condition) {
  const conditionMatches = filterByCondition(clubs, condition);
  const affordable = filterByBudget(conditionMatches, budget);
  const selected = selectForDisplay(affordable);
  countElement.textContent = `${selected.length} ${selected.length === 1 ? "club" : "clubs"}`;

  if (!selected.length) {
    const message =
      condition === "used"
        ? `No clubs match this budget and used filter. Used means ${USED_MAX_YEAR} or older.`
        : "No clubs match this budget.";
    listElement.innerHTML = `<div class="notice">${escapeHtml(message)}</div>`;
    return;
  }

  listElement.innerHTML = selected
    .map((club) => {
      const meta = [
        club.year ? `${club.year} model` : null,
        typeof club.msrp === "number"
          ? `MSRP ${currency.format(club.msrp)}`
          : null,
      ]
        .filter(Boolean)
        .join(" - ");
      const reasons = club.reasons
        .slice(0, 3)
        .map((reason) => `<li>${escapeHtml(reason)}</li>`)
        .join("");

      return `
        <article class="club-card ${scoreBand(club.score)}">
          <div>
            <div class="club-title">
              <h4>${escapeHtml(club.name)}</h4>
              <span class="club-meta">${escapeHtml(meta)}</span>
            </div>
          </div>
          <div class="score-badge ${scoreBand(club.score)}">${club.score}%</div>
          <ul class="reason-list">${reasons}</ul>
        </article>
      `;
    })
    .join("");
}

function scoreBand(score) {
  if (score >= 85) return "band-high";
  if (score >= 65) return "band-mid";
  return "band-low";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-button").forEach((button) => {
    const selected = button.dataset.tab === tabName;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    // Roving tabindex: only the selected tab is a tab stop.
    button.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  });
}

function updateTabAvailability(result) {
  const driverTab = document.querySelector('[data-tab="drivers"]');
  const ironTab = document.querySelector('[data-tab="irons"]');
  driverTab.classList.toggle("hidden", !result.wants_driver);
  ironTab.classList.toggle("hidden", !result.wants_irons);

  if (!result.wants_driver && state.activeTab === "drivers") {
    setActiveTab("irons");
  } else if (!result.wants_irons && state.activeTab === "irons") {
    setActiveTab("drivers");
  } else if (result.wants_driver) {
    setActiveTab("drivers");
  } else {
    setActiveTab("irons");
  }

  tabs.classList.toggle("hidden", !(result.wants_driver && result.wants_irons));
}

function renderRecommendations() {
  if (!state.result) {
    return;
  }

  document.querySelector("#driver-budget-value").textContent = currency.format(
    driverBudget.value,
  );
  document.querySelector("#iron-budget-value").textContent = currency.format(
    ironBudget.value,
  );

  renderClubList(
    document.querySelector("#driver-list"),
    document.querySelector("#driver-count"),
    state.result.recommendations.drivers,
    Number(driverBudget.value),
    driverCondition.value,
  );
  renderClubList(
    document.querySelector("#iron-list"),
    document.querySelector("#iron-count"),
    state.result.recommendations.irons,
    Number(ironBudget.value),
    ironCondition.value,
  );
}

// A focused number input treats the wheel as increment/decrement, so scrolling
// the page over one silently rewrites the handicap or swing speed. Dropping
// focus stops the edit and leaves the scroll itself untouched.
document.addEventListener(
  "wheel",
  (event) => {
    const active = document.activeElement;
    if (active && active.type === "number" && active === event.target) {
      active.blur();
    }
  },
  { passive: true },
);

form.addEventListener("change", updateConditionalFields);
driverBudget.addEventListener("input", renderRecommendations);
ironBudget.addEventListener("input", renderRecommendations);
driverCondition.addEventListener("change", () => {
  if (!state.result) {
    return;
  }
  updateBudgetForCondition(
    driverBudget,
    state.result.recommendations.drivers,
    driverCondition.value,
    2500,
  );
  renderRecommendations();
});
ironCondition.addEventListener("change", () => {
  if (!state.result) {
    return;
  }
  updateBudgetForCondition(
    ironBudget,
    state.result.recommendations.irons,
    ironCondition.value,
    2500,
  );
  renderRecommendations();
});

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

// Left/Right/Home/End move between tabs, per the ARIA tabs pattern.
tabs.addEventListener("keydown", (event) => {
  const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
  if (!keys.includes(event.key)) {
    return;
  }
  const available = [...document.querySelectorAll(".tab-button")].filter(
    (button) => !button.classList.contains("hidden"),
  );
  if (available.length < 2) {
    return;
  }
  event.preventDefault();
  const current = available.findIndex((button) => button.tabIndex === 0);
  let next = current < 0 ? 0 : current;
  if (event.key === "ArrowLeft") {
    next = (current - 1 + available.length) % available.length;
  } else if (event.key === "ArrowRight") {
    next = (current + 1) % available.length;
  } else if (event.key === "Home") {
    next = 0;
  } else {
    next = available.length - 1;
  }
  setActiveTab(available[next].dataset.tab);
  available[next].focus();
});

function revealResults() {
  // On narrow screens the form stacks above the results, leaving the outcome a
  // full screen below the fold; without this, submitting looks like a no-op.
  const panel = document.querySelector(".results-panel");
  if (!panel) {
    return;
  }
  const { top } = panel.getBoundingClientRect();
  const alreadyInView = top >= 0 && top < window.innerHeight * 0.5;
  if (alreadyInView) {
    return;
  }
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  panel.scrollIntoView({
    behavior: reduceMotion ? "auto" : "smooth",
    block: "start",
  });
}

function announceResults() {
  const counts = [];
  if (state.result.wants_driver) {
    counts.push(document.querySelector("#driver-count").textContent.trim());
  }
  if (state.result.wants_irons) {
    counts.push(document.querySelector("#iron-count").textContent.trim());
  }
  resultStatus.textContent = counts.length
    ? `Recommendations ready: ${counts.join(" and ")}.`
    : "Recommendations ready.";
}

async function requestRecommendations() {
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.setAttribute("aria-busy", "true");
  formError.classList.add("hidden");
  emptyState.classList.add("hidden");
  recommendations.classList.add("hidden");
  loadingState.classList.remove("hidden");
  resultStatus.textContent = "Running the model.";
  // Move to the panel now so the spinner, and then the outcome, are on screen.
  revealResults();

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
    state.result = await response.json();
    driverCondition.value = "all";
    ironCondition.value = "all";
    setBudgetStart(driverBudget, state.result.recommendations.drivers, 2500);
    setBudgetStart(ironBudget, state.result.recommendations.irons, 2500);
    updateSpecGrid(state.result.specs, state.result);
    updateTabAvailability(state.result);
    loadingState.classList.add("hidden");
    recommendations.classList.remove("hidden");
    renderRecommendations();
    announceResults();
  } catch (error) {
    console.error(error);
    loadingState.classList.add("hidden");
    formErrorDetail.textContent =
      error instanceof TypeError
        ? "The server didn't respond. Check that it's still running, then try again."
        : `${error.message}. Try again, or adjust the profile and rerun.`;
    formError.classList.remove("hidden");
    // Results are gone, so put the starting instruction back on screen.
    if (!state.result) {
      emptyState.classList.remove("hidden");
    } else {
      recommendations.classList.remove("hidden");
    }
    retryAction.focus();
  } finally {
    submitButton.disabled = false;
    submitButton.removeAttribute("aria-busy");
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  requestRecommendations();
});

retryAction.addEventListener("click", requestRecommendations);

updateConditionalFields();
