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
  tiles[2].textContent = result.wants_irons ? titleCaseSpec(specs.iron_category) : "-";
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
    .forEach((section) => section.classList.toggle("hidden", shoppingFor === "Irons"));
  document
    .querySelectorAll(".iron-fields")
    .forEach((section) => section.classList.toggle("hidden", shoppingFor === "Driver"));
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
  return clubs.filter((club) => Number.isInteger(club.year) && club.year <= USED_MAX_YEAR);
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
        typeof club.msrp === "number" ? `MSRP ${currency.format(club.msrp)}` : null,
      ]
        .filter(Boolean)
        .join(" - ");
      const reasons = club.reasons
        .slice(0, 3)
        .map((reason) => `<li>${escapeHtml(reason)}</li>`)
        .join("");

      return `
        <article class="club-card">
          <div>
            <div class="club-title">
              <h4>${escapeHtml(club.name)}</h4>
              <span class="club-meta">${escapeHtml(meta)}</span>
            </div>
          </div>
          <div class="score-badge">${club.score}%</div>
          <ul class="reason-list">${reasons}</ul>
        </article>
      `;
    })
    .join("");
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
    button.classList.toggle("active", button.dataset.tab === tabName);
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

  document.querySelector("#driver-budget-value").textContent = currency.format(driverBudget.value);
  document.querySelector("#iron-budget-value").textContent = currency.format(ironBudget.value);

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
  updateBudgetForCondition(ironBudget, state.result.recommendations.irons, ironCondition.value, 2500);
  renderRecommendations();
});

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!response.ok) {
      throw new Error("Recommendation request failed");
    }
    state.result = await response.json();
    driverCondition.value = "all";
    ironCondition.value = "all";
    setBudgetStart(driverBudget, state.result.recommendations.drivers, 2500);
    setBudgetStart(ironBudget, state.result.recommendations.irons, 2500);
    updateSpecGrid(state.result.specs, state.result);
    updateTabAvailability(state.result);
    emptyState.classList.add("hidden");
    recommendations.classList.remove("hidden");
    renderRecommendations();
  } catch (error) {
    console.error(error);
    alert("Unable to generate recommendations. Please try again.");
  } finally {
    submitButton.disabled = false;
  }
});

updateConditionalFields();
