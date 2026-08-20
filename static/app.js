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
const USED_MAX_YEAR = new Date().getFullYear() - 4;

// One entry per results category. Driver and Fairway Wood share a profile
// section (same swing, same session) so two tabs point at the same
// fieldsSelector; every other lookup - result JSON key, "wants" flag, element
// id prefix, starting budget ceiling - is per-tab because those genuinely
// differ per category.
const CATEGORIES = [
  { tab: "drivers", resultKey: "drivers", wants: "wants_driver", idPrefix: "driver", fieldsSelector: ".wood-fields", fallbackBudget: 2500 },
  { tab: "fairway-woods", resultKey: "fairway_woods", wants: "wants_fairway_woods", idPrefix: "fairway-wood", fieldsSelector: ".wood-fields", fallbackBudget: 500 },
  { tab: "irons", resultKey: "irons", wants: "wants_irons", idPrefix: "iron", fieldsSelector: ".iron-fields", fallbackBudget: 2500 },
  { tab: "wedges", resultKey: "wedges", wants: "wants_wedges", idPrefix: "wedge", fieldsSelector: ".wedge-fields", fallbackBudget: 350 },
];

const categories = CATEGORIES.map((category) => ({
  ...category,
  budgetEl: document.querySelector(`#${category.idPrefix}-budget`),
  budgetValueEl: document.querySelector(`#${category.idPrefix}-budget-value`),
  conditionEl: document.querySelector(`#${category.idPrefix}-condition`),
  listEl: document.querySelector(`#${category.idPrefix}-list`),
  countEl: document.querySelector(`#${category.idPrefix}-count`),
}));

const SPEC_TILES = [
  { spec: "driver_loft", wants: "wants_driver", format: (value) => `${value} deg` },
  { spec: "shaft_flex", wants: "wants_driver", format: (value) => value },
  { spec: "fairway_wood_loft", wants: "wants_fairway_woods", format: (value) => `${value} deg` },
  { spec: "iron_category", wants: "wants_irons", format: (value) => titleCaseSpec(value) },
  { spec: "wedge_bounce", wants: "wants_wedges", format: (value) => value },
];

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
  SPEC_TILES.forEach(({ spec, wants, format }) => {
    const tile = document.querySelector(`.spec-tile[data-spec="${spec}"] strong`);
    tile.textContent = result[wants] ? format(specs[spec]) : "-";
  });
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
  // Driver and Fairway Wood share .wood-fields (same swing, same inputs), so
  // that section shows for either; a category with its own question section
  // only appears once its own checkbox is checked.
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

function displayYears(club, condition) {
  const years =
    Array.isArray(club.years) && club.years.length
      ? club.years
      : club.year
        ? [club.year]
        : [];
  if (condition === "used") {
    return years.filter((year) => year <= USED_MAX_YEAR);
  }
  return years;
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

  const topScore = selected[0].score;
  const tiedCount = selected.filter((club) => club.score === topScore).length;
  // A tie means the model genuinely can't separate these clubs - many real
  // clubs suit the same golfer equally well - so say so instead of letting
  // list position imply one beats the others.
  const tieNote =
    tiedCount > 1
      ? `<div class="tie-note">${tiedCount} clubs are an equally strong fit at ${topScore}% - the order below doesn't mean one beats the others.</div>`
      : "";

  listElement.innerHTML = tieNote + selected
    .map((club) => {
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
          <div>
            <div class="club-title">
              <h4>${escapeHtml(club.name)}</h4>
              <span class="club-meta">${escapeHtml(meta)}</span>
            </div>
          </div>
          <div class="score-block">
            <span class="score-label">Stickly Score</span>
            <div class="score-badge ${scoreBand(club.score)}">${club.score}%</div>
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

  // No point showing tabs to switch between when at most one has anything in it.
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

// Unchecking the last box would leave nothing to shop for and no visible tab
// to recover from, so the group always keeps at least one category checked.
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
  const counts = categories
    .filter((category) => state.result[category.wants])
    .map((category) => category.countEl.textContent.trim());
  resultStatus.textContent = counts.length
    ? `Recommendations ready: ${counts.join(", ")}.`
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
    categories.forEach((category) => {
      category.conditionEl.value = "all";
      setBudgetStart(
        category.budgetEl,
        state.result.recommendations[category.resultKey],
        category.fallbackBudget,
      );
    });
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

/* ---------- Mobile profile drawer ----------
   Below 920px the profile panel is an off-canvas overlay rather than a column
   (see styles.css). The class on <body> is what the slide and the scroll lock
   are keyed off, so what's left here is that state plus the focus handling any
   overlay owes the keyboard. */

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
  // The close button rather than the first field: it sits at the top of the
  // panel, so focusing it can't scroll the form before it's been seen.
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
  // An overlay shouldn't leak the keyboard into the results sitting behind the
  // scrim, so Tab wraps at both ends of the panel while it's open.
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

// Past the breakpoint the panel is a plain sidebar again and the CSS follows
// the media query by itself - but aria-expanded and the scroll lock hang off
// the class, so a drawer left open while widening has to be cleared.
drawerBreakpoint.addEventListener("change", (event) => {
  if (!event.matches) {
    setDrawer(false);
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  // The drawer covers the results on mobile, so leaving it open would hide the
  // spinner and then the outcome behind it. Focus stays where the submit left
  // it rather than snapping back to the toggle.
  closeDrawer({ returnFocus: false });
  requestRecommendations();
});

retryAction.addEventListener("click", requestRecommendations);

// "Current driver"/"current irons" are typeahead text fields backed by a
// <datalist> of real catalog entries; collectPayload() reads the hidden id
// field, not the visible text. The <datalist> is a data source only (no
// list="" attribute links it to the input) - suggestions are rendered as a
// custom dropdown instead, because native datalist popups are entirely
// browser/OS-controlled and unstylable: on mobile several browsers render
// them as an autofill strip glued to the keyboard rather than a list anchored
// to the field, which reads as broken. Typing something that doesn't exactly
// match a real option (case-insensitive) leaves the hidden id empty - same
// as picking "skip" - rather than guessing at a fuzzy match.
function wireCatalogTypeahead(textFieldName, datalistId, hiddenFieldName, listboxId) {
  const textField = form.querySelector(`[name="${textFieldName}"]`);
  const datalist = document.querySelector(`#${datalistId}`);
  const hiddenField = form.querySelector(`[name="${hiddenFieldName}"]`);
  const listbox = document.querySelector(`#${listboxId}`);
  if (!textField || !datalist || !hiddenField || !listbox) return;

  // Moved out of .profile-panel (which scrolls) to a direct child of <body>,
  // so it renders relative to the viewport instead of a scrolling ancestor
  // that could clip it. Position is computed from the field's own on-screen
  // rect (positionList, below) rather than normal document flow.
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
    // capture: true to match the addEventListener call in renderMatches -
    // scroll events don't bubble, so catching them from .profile-panel's
    // internal scroll (or any other scrolling ancestor) requires listening
    // during the capture phase, not the default bubble phase.
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

  // mousedown, not click: it fires before the input's blur, so the list is
  // still open (and the tapped option still exists) when we act on it - a
  // click handler here would fire after blur already closed the list.
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

updateConditionalFields();
