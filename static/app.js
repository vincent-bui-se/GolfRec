import { USED_MAX_YEAR, selectForDisplay, filterByCondition, displayYears, filterByBudget, maxMsrp } from "./filters.mjs";

const state = {
  result: null,
  activeTab: "drivers",
};

const form = document.querySelector("#profile-form");
const emptyState = document.querySelector("#empty-state");
const recommendations = document.querySelector("#recommendations");
const tabs = document.querySelector("#tabs");
const loadingState = document.querySelector("#loading-state");
const formError = document.querySelector("#form-error");
const formErrorDetail = document.querySelector("#form-error-detail");
const retryAction = document.querySelector("#retry-action");
const resultStatus = document.querySelector("#result-status");
const specGrid = document.querySelector("#spec-grid");
const fitBasis = document.querySelector("#fit-basis");
const savedNote = document.querySelector("#saved-note");

const CATEGORIES = [
  { tab: "drivers", resultKey: "drivers", wants: "wants_driver", idPrefix: "driver", fallbackBudget: 2500 },
  { tab: "fairway-woods", resultKey: "fairway_woods", wants: "wants_fairway_woods", idPrefix: "fairway-wood", fallbackBudget: 500 },
  { tab: "irons", resultKey: "irons", wants: "wants_irons", idPrefix: "iron", fallbackBudget: 2500 },
  { tab: "wedges", resultKey: "wedges", wants: "wants_wedges", idPrefix: "wedge", fallbackBudget: 350 },
];

const categories = CATEGORIES.map((category) => ({
  ...category,
  budgetEl: document.querySelector(`#${category.idPrefix}-budget`),
  budgetValueEl: document.querySelector(`#${category.idPrefix}-budget-value`),
  conditionEl: document.querySelector(`#${category.idPrefix}-condition`),
  listEl: document.querySelector(`#${category.idPrefix}-list`),
  countEl: document.querySelector(`#${category.idPrefix}-count`),
}));

const FLEX_NAMES = { L: "Ladies", A: "Senior", R: "Regular", S: "Stiff", X: "Extra stiff" };

const SPEC_TILES = [
  { spec: "driver_loft", wants: ["wants_driver"], format: (value) => `${value}°` },
  // There's no separate iron flex model yet, so woods and irons share one.
  {
    spec: "shaft_flex",
    wants: ["wants_driver", "wants_fairway_woods", "wants_irons"],
    format: (value) => (FLEX_NAMES[value] ? `${FLEX_NAMES[value]} (${value})` : value),
  },
  { spec: "fairway_wood_loft", wants: ["wants_fairway_woods"], format: (value) => `${value}°` },
  { spec: "iron_category", wants: ["wants_irons"], format: (value) => titleCaseSpec(value) },
  { spec: "wedge_bounce", wants: ["wants_wedges"], format: (value) => value },
];

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function formValue(name) {
  return form.elements[name]?.value ?? "";
}

function checkedShoppingFor() {
  return [...form.querySelectorAll('input[name="shopping_for"]:checked')].map(
    (checkbox) => checkbox.value,
  );
}

function collectPayload() {
  return {
    shopping_for: checkedShoppingFor(),
    score_mode: formValue("score_mode"),
    handicap: Number(formValue("handicap")),
    average_score: Number(formValue("average_score")),
    speed_mode: formValue("speed_mode"),
    swing_speed: Number(formValue("swing_speed")),
    driver_carry: Number(formValue("driver_carry")),
    swing_tempo: formValue("swing_tempo"),
    current_driver_id: formValue("current_driver_id"),
    driver_shot_shape: formValue("driver_shot_shape"),
    driver_trajectory: formValue("driver_trajectory"),
    driver_goal: formValue("driver_goal"),
    current_iron_set_id: formValue("current_iron_set_id"),
    iron_shot_shape: formValue("iron_shot_shape"),
    iron_goal: formValue("iron_goal"),
    iron_trajectory: formValue("iron_trajectory"),
    iron_feel: formValue("iron_feel"),
    iron_miss: formValue("iron_miss"),
    wedge_turf: formValue("wedge_turf"),
    divot_depth: formValue("divot_depth"),
    wedge_shot_style: formValue("wedge_shot_style"),
    wedge_loft: formValue("wedge_loft"),
  };
}

function titleCaseSpec(value) {
  return String(value || "-")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function updateSpecGrid(specs, result) {
  const notes = result.spec_notes || {};
  SPEC_TILES.forEach(({ spec, wants, format }) => {
    const tile = document.querySelector(`.spec-tile[data-spec="${spec}"]`);
    const value = tile.querySelector("strong");
    const note = tile.querySelector(".spec-note");
    const wanted = wants.some((key) => result[key]);
    tile.classList.toggle("spec-off", !wanted);
    note.textContent = wanted ? notes[spec] || "" : "Not shopping for this";
    const next = wanted ? format(specs[spec]) : "-";
    if (value.textContent === next) {
      return;
    }
    value.textContent = next;
    // Force a reflow so the highlight animation restarts.
    tile.classList.remove("spec-changed");
    void tile.offsetWidth;
    tile.classList.add("spec-changed");
  });
}

// "an 8", "an 18", "an 85" (same rule as with_article in recommend.py).
function withArticle(number) {
  const text = String(number);
  return /^(8\d?|11|18)(\D|$)/.test(text) ? `an ${text}` : `a ${text}`;
}

function describeFitBasis(profile) {
  const speedMph = Math.round(profile.swing_speed);
  const carry = Math.round(profile.driver_carry);
  const handicap =
    profile.handicap_source === "average score"
      ? `${withArticle(profile.handicap)} handicap (estimated from your average score)`
      : `${withArticle(profile.handicap)} handicap`;
  const speed =
    profile.speed_source === "carry"
      ? `${withArticle(speedMph)} mph swing (estimated from ${carry} yards of carry)`
      : `${withArticle(speedMph)} mph swing, about ${carry} yards of carry`;
  return `Fitted for ${handicap} and ${speed}.`;
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

  const checked = new Set(checkedShoppingFor());
  document.querySelectorAll(".speed-fields").forEach((section) =>
    section.classList.toggle(
      "hidden",
      !checked.has("Driver") && !checked.has("Fairway Wood") && !checked.has("Irons"),
    ),
  );
  document.querySelectorAll(".wood-fields").forEach((section) =>
    section.classList.toggle("hidden", !checked.has("Driver") && !checked.has("Fairway Wood")),
  );
  document.querySelectorAll(".iron-fields").forEach((section) =>
    section.classList.toggle("hidden", !checked.has("Irons")),
  );
  document.querySelectorAll(".wedge-fields").forEach((section) =>
    section.classList.toggle("hidden", !checked.has("Wedges")),
  );
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
  countElement.textContent = `${selected.length} ${selected.length === 1 ? "club" : "clubs"} · fit score out of 100`;

  if (!selected.length) {
    // Tell the user how far to raise the budget.
    const cheapest = conditionMatches
      .map((club) => club.msrp)
      .filter((price) => typeof price === "number" && Number.isFinite(price))
      .sort((a, b) => a - b)[0];

    const cause =
      condition === "used"
        ? `No clubs match this budget with Used selected. Used means ${USED_MAX_YEAR} or older.`
        : "No clubs match this budget.";
    const recovery =
      typeof cheapest === "number"
        ? ` The cheapest that fits your other answers is $${Math.round(cheapest)}.`
        : "";

    listElement.innerHTML = `<div class="notice">${escapeHtml(cause + recovery)}</div>`;
    return;
  }

  const topScore = selected[0].score;
  // Scores within 2 points are basically a tie given how coarse the catalog data is.
  const NEAR_TIE_THRESHOLD = 2;
  const closeGroup = selected.filter((club) => topScore - club.score <= NEAR_TIE_THRESHOLD);
  const exactTie = closeGroup.length > 1 && closeGroup.every((club) => club.score === topScore);
  let tieNote = "";
  if (exactTie) {
    tieNote = `<div class="tie-note">${closeGroup.length} clubs tie at ${topScore}. The order below isn't a ranking, so choose on the reasons and the price.</div>`;
  } else if (closeGroup.length > 1) {
    const lowScore = closeGroup[closeGroup.length - 1].score;
    tieNote = `<div class="tie-note">The top ${closeGroup.length} land within ${NEAR_TIE_THRESHOLD} points (${lowScore}-${topScore}). That gap isn't a meaningful difference, so go by the reasons rather than the number.</div>`;
  }

  listElement.innerHTML = tieNote + selected
    .map((club, index) => {
      const years = displayYears(club, condition);
      const meta = [
        years.length > 1
          ? `${years.join("/")} models`
          : years.length === 1
            ? `${years[0]} model`
            : null,
        typeof club.msrp === "number"
          ? years.length > 1
            ? `MSRP from ${currency.format(club.msrp)}`
            : `MSRP ${currency.format(club.msrp)}`
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
          <div class="club-title">
            <span class="club-rank" aria-hidden="true">${index + 1}</span>
            <h4>${escapeHtml(club.name)}</h4>
            <span class="club-meta">${escapeHtml(meta)}</span>
          </div>
          <div class="score-block" role="img" aria-label="Fit score ${club.score} out of 100">
            <div class="score-badge ${scoreBand(club.score)}">${club.score}</div>
          </div>
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
  const visibleTabs = [];
  categories.forEach((category) => {
    const wanted = Boolean(result[category.wants]);
    document.querySelector(`[data-tab="${category.tab}"]`).classList.toggle("hidden", !wanted);
    if (wanted) {
      visibleTabs.push(category.tab);
    }
  });

  if (!visibleTabs.includes(state.activeTab)) {
    setActiveTab(visibleTabs[0] || categories[0].tab);
  }

  tabs.classList.toggle("hidden", visibleTabs.length < 2);
}

function renderRecommendations() {
  if (!state.result) {
    return;
  }

  categories.forEach((category) => {
    category.budgetValueEl.textContent = currency.format(category.budgetEl.value);
    renderClubList(
      category.listEl,
      category.countEl,
      state.result.recommendations[category.resultKey],
      Number(category.budgetEl.value),
      category.conditionEl.value,
    );
  });
}

// Stop the mouse wheel from changing a focused number input while scrolling.
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

// Remember the form between visits (localStorage only).

const STORAGE_KEY = "stickly.profile.v1";

function saveAnswers() {
  try {
    const answers = {};
    [...form.elements].forEach((field) => {
      if (!field.name) return;
      if (field.type === "checkbox") {
        answers[field.name] = answers[field.name] || [];
        if (field.checked) answers[field.name].push(field.value);
      } else if (field.type === "radio") {
        if (field.checked) answers[field.name] = field.value;
      } else {
        answers[field.name] = field.value;
      }
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
    savedNote.classList.remove("hidden");
  } catch {
    // Storage can be blocked (private mode); the form still works without it.
  }
}

function restoreAnswers() {
  let answers;
  try {
    answers = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
  } catch {
    answers = null;
  }
  if (!answers || typeof answers !== "object") return;
  [...form.elements].forEach((field) => {
    if (!field.name || !(field.name in answers)) return;
    const saved = answers[field.name];
    if (field.type === "checkbox") {
      field.checked = Array.isArray(saved) && saved.includes(field.value);
    } else if (field.type === "radio") {
      field.checked = field.value === saved;
    } else if (field.tagName === "SELECT") {
      if ([...field.options].some((option) => option.value === saved)) field.value = saved;
    } else {
      field.value = String(saved);
    }
  });
  if (!checkedShoppingFor().length) {
    form.querySelectorAll('input[name="shopping_for"]').forEach((box) => {
      box.checked = true;
    });
  }
  savedNote.classList.remove("hidden");
}

document.querySelector("#reset-answers").addEventListener("click", () => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore blocked storage.
  }
  form.reset();
  form.querySelectorAll('input[type="hidden"]').forEach((field) => {
    field.value = "";
  });
  savedNote.classList.add("hidden");
  updateConditionalFields();
});

// Keep at least one category checked.
document.querySelectorAll('input[name="shopping_for"]').forEach((checkbox) => {
  checkbox.addEventListener("click", (event) => {
    if (checkedShoppingFor().length === 0) {
      event.target.checked = true;
    }
  });
});

categories.forEach((category) => {
  category.budgetEl.addEventListener("input", renderRecommendations);
  category.conditionEl.addEventListener("change", () => {
    if (!state.result) {
      return;
    }
    updateBudgetForCondition(
      category.budgetEl,
      state.result.recommendations[category.resultKey],
      category.conditionEl.value,
      category.fallbackBudget,
    );
    renderRecommendations();
  });
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
  // On mobile the results can be off screen after submitting.
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
  const counts = categories
    .filter((category) => state.result[category.wants])
    .map((category) => `${category.countEl.textContent.split(" · ")[0]} for ${category.tab.replace("-", " ")}`);
  resultStatus.textContent = counts.length
    ? `Recommendations ready: ${counts.join(", ")}.`
    : "Recommendations ready.";
}

function describeFailure(error) {
  if (error instanceof TypeError) {
    return "Couldn't reach Stickly. Check your connection, then try again.";
  }
  if (error.status >= 500) {
    return "Stickly had a problem scoring your profile. Try again in a moment.";
  }
  if (error.status >= 400) {
    return error.publicMessage || "Something in the profile didn't come through. Change an answer, then run it again.";
  }
  return "Something went wrong. Try again in a moment.";
}

async function requestRecommendations() {
  const submitButton = form.querySelector("button[type='submit']");
  submitButton.disabled = true;
  submitButton.setAttribute("aria-busy", "true");
  formError.classList.add("hidden");
  emptyState.classList.add("hidden");
  recommendations.classList.add("hidden");
  loadingState.classList.remove("hidden");
  resultStatus.textContent = "Fitting your clubs.";
  revealResults();

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    if (!response.ok) {
      const failure = new Error(`Server responded with ${response.status}`);
      failure.status = response.status;
      if (response.status === 400) {
        const detail = await response.json().catch(() => null);
        failure.publicMessage = detail?.error;
      }
      throw failure;
    }
    state.result = await response.json();
    categories.forEach((category) => {
      category.conditionEl.value = "all";
      setBudgetStart(
        category.budgetEl,
        state.result.recommendations[category.resultKey],
        category.fallbackBudget,
      );
    });
    updateSpecGrid(state.result.specs, state.result);
    specGrid.classList.remove("hidden");
    fitBasis.textContent = describeFitBasis(state.result.profile);
    fitBasis.classList.remove("hidden");
    updateTabAvailability(state.result);
    loadingState.classList.add("hidden");
    recommendations.classList.remove("hidden");
    renderRecommendations();
    announceResults();
  } catch (error) {
    console.error(error);
    loadingState.classList.add("hidden");
    formErrorDetail.textContent = describeFailure(error);
    formError.classList.remove("hidden");
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

// Mobile profile drawer

const drawerToggle = document.querySelector("#drawer-toggle");
const drawerClose = document.querySelector("#drawer-close");
const drawerScrim = document.querySelector("#drawer-scrim");
const profilePanel = document.querySelector("#profile-panel");
const drawerBreakpoint = window.matchMedia("(max-width: 920px)");

function drawerIsOpen() {
  return document.body.classList.contains("drawer-open");
}

function setDrawer(open) {
  document.body.classList.toggle("drawer-open", open);
  drawerToggle.setAttribute("aria-expanded", String(open));
}

function openDrawer() {
  setDrawer(true);
  drawerClose.focus();
}

function closeDrawer({ returnFocus = true } = {}) {
  if (!drawerIsOpen()) {
    return;
  }
  setDrawer(false);
  if (returnFocus) {
    drawerToggle.focus();
  }
}

function focusablesInDrawer() {
  return Array.from(
    profilePanel.querySelectorAll("button, input, select, textarea, [href]"),
  ).filter((element) => !element.disabled && element.offsetParent !== null);
}

drawerToggle.addEventListener("click", () => {
  if (drawerIsOpen()) {
    closeDrawer();
  } else {
    openDrawer();
  }
});

drawerClose.addEventListener("click", () => closeDrawer());
drawerScrim.addEventListener("click", () => closeDrawer());

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawer();
    return;
  }
  // Trap focus inside the open drawer.
  if (event.key !== "Tab" || !drawerIsOpen()) {
    return;
  }
  const reachable = focusablesInDrawer();
  if (!reachable.length) {
    return;
  }
  const first = reachable[0];
  const last = reachable[reachable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

// Reset drawer state when the window widens past the breakpoint.
drawerBreakpoint.addEventListener("change", (event) => {
  if (!event.matches) {
    setDrawer(false);
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  saveAnswers();
  closeDrawer({ returnFocus: false });
  requestRecommendations();
});

retryAction.addEventListener("click", requestRecommendations);

document.querySelector("#run-defaults").addEventListener("click", () => form.requestSubmit());
document.querySelector("#open-profile").addEventListener("click", openDrawer);

// Typeahead for the current driver/irons. An exact match fills the hidden id field.
function wireCatalogTypeahead(textFieldName, datalistId, hiddenFieldName, listboxId) {
  const textField = form.querySelector(`[name="${textFieldName}"]`);
  const datalist = document.querySelector(`#${datalistId}`);
  const hiddenField = form.querySelector(`[name="${hiddenFieldName}"]`);
  const listbox = document.querySelector(`#${listboxId}`);
  if (!textField || !datalist || !hiddenField || !listbox) return;

  // Move to <body> so the scrolling sidebar doesn't clip it.
  document.body.appendChild(listbox);

  const options = [...datalist.options].map((option) => ({
    id: option.dataset.id,
    label: option.value,
  }));

  let visibleMatches = [];
  let activeIndex = -1;

  function positionList() {
    const rect = textField.getBoundingClientRect();
    listbox.style.top = `${rect.bottom + 4}px`;
    listbox.style.left = `${rect.left}px`;
    listbox.style.width = `${rect.width}px`;
  }

  function resolveHiddenId() {
    const typed = textField.value.trim().toLowerCase();
    const match = options.find((option) => option.label.toLowerCase() === typed);
    hiddenField.value = match ? match.id : "";
  }

  function closeList() {
    listbox.hidden = true;
    listbox.innerHTML = "";
    textField.setAttribute("aria-expanded", "false");
    textField.removeAttribute("aria-activedescendant");
    visibleMatches = [];
    activeIndex = -1;
    // Capture, because scroll events from the sidebar don't bubble.
    window.removeEventListener("scroll", positionList, true);
    window.removeEventListener("resize", positionList);
  }

  function setActive(index) {
    activeIndex = index;
    [...listbox.children].forEach((li, i) => {
      li.setAttribute("aria-selected", String(i === activeIndex));
    });
    const active = listbox.children[activeIndex];
    if (active) {
      active.scrollIntoView({ block: "nearest" });
      textField.setAttribute("aria-activedescendant", active.id);
    } else {
      textField.removeAttribute("aria-activedescendant");
    }
  }

  function selectMatch(option) {
    textField.value = option.label;
    hiddenField.value = option.id;
    closeList();
  }

  function renderMatches(typed) {
    visibleMatches = options
      .filter((option) => option.label.toLowerCase().includes(typed))
      .slice(0, 8);
    listbox.innerHTML = visibleMatches
      .map(
        (option, index) =>
          `<li role="option" id="${listboxId}-opt-${index}" data-index="${index}">${escapeHtml(option.label)}</li>`,
      )
      .join("");
    const hasMatches = visibleMatches.length > 0;
    listbox.hidden = !hasMatches;
    textField.setAttribute("aria-expanded", String(hasMatches));
    activeIndex = -1;
    if (hasMatches) {
      positionList();
      window.addEventListener("scroll", positionList, true);
      window.addEventListener("resize", positionList);
    }
  }

  textField.addEventListener("input", () => {
    resolveHiddenId();
    const typed = textField.value.trim().toLowerCase();
    if (!typed) {
      closeList();
      return;
    }
    renderMatches(typed);
  });

  textField.addEventListener("focus", () => {
    const typed = textField.value.trim().toLowerCase();
    if (typed) renderMatches(typed);
  });

  textField.addEventListener("keydown", (event) => {
    if (listbox.hidden) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive(Math.min(activeIndex + 1, visibleMatches.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive(Math.max(activeIndex - 1, 0));
    } else if (event.key === "Enter" && activeIndex >= 0) {
      event.preventDefault();
      selectMatch(visibleMatches[activeIndex]);
    } else if (event.key === "Escape") {
      closeList();
    }
  });

  // mousedown fires before the input's blur closes the list; click would be too late.
  listbox.addEventListener("mousedown", (event) => {
    const li = event.target.closest("li");
    if (!li) return;
    event.preventDefault();
    const option = visibleMatches[Number(li.dataset.index)];
    if (option) selectMatch(option);
  });

  textField.addEventListener("blur", () => {
    setTimeout(closeList, 0);
  });
}

wireCatalogTypeahead(
  "current_driver_label",
  "current-driver-options",
  "current_driver_id",
  "current-driver-listbox",
);
wireCatalogTypeahead(
  "current_iron_set_label",
  "current-iron-set-options",
  "current_iron_set_id",
  "current-iron-set-listbox",
);

document.querySelectorAll(".info-icon[data-tooltip]").forEach((icon) => {
  const withYear = `Used shows clubs at least four model years old, so ${USED_MAX_YEAR} or older.`;
  icon.setAttribute("aria-label", withYear);
  icon.dataset.tooltip = withYear;
});

restoreAnswers();
updateConditionalFields();
