import assert from "node:assert/strict";
import test from "node:test";
import {
  USED_MAX_YEAR, displayYears, filterByBudget, filterByCondition,
  maxMsrp, selectForDisplay,
} from "../static/filters.mjs";

test("budget filtering excludes unknown prices and orders by score", () => {
  const clubs = [
    { name: "Affordable", msrp: 400, score: 80 },
    { name: "Best fit", msrp: 600, score: 90 },
    { name: "Too expensive", msrp: 601, score: 95 },
    { name: "Unknown price", msrp: null, score: 99 },
  ];
  assert.deepEqual(filterByBudget(clubs, 600).map((c) => c.name), ["Best fit", "Affordable"]);
});

test("budget filtering keeps the server's price order when scores tie", () => {
  const clubs = [{ msrp: 300, score: 90 }, { msrp: 500, score: 90 }];
  assert.deepEqual(filterByBudget(clubs, 600), clubs);
});

test("used condition applies the four-model-year cutoff", () => {
  const older = { year: USED_MAX_YEAR - 1 };
  const cutoff = { year: USED_MAX_YEAR };
  const newer = { year: USED_MAX_YEAR + 1 };
  const clubs = [older, cutoff, newer, { year: null }];
  assert.deepEqual(filterByCondition(clubs, "used"), [older, cutoff]);
  assert.deepEqual(filterByCondition(clubs, "all"), clubs);
});

test("merged models show only eligible years with used selected", () => {
  const club = { year: USED_MAX_YEAR, years: [USED_MAX_YEAR, USED_MAX_YEAR + 1] };
  assert.deepEqual(displayYears(club, "used"), [USED_MAX_YEAR]);
  assert.deepEqual(displayYears(club, "all"), club.years);
  assert.deepEqual(displayYears({ year: USED_MAX_YEAR }, "used"), [USED_MAX_YEAR]);
  assert.deepEqual(displayYears({}, "all"), []);
});

test("display expands a flat top five through the first lower score", () => {
  const clubs = [91, 91, 91, 91, 91, 91, 89, 88].map((score) => ({ score }));
  assert.deepEqual(selectForDisplay(clubs).map((c) => c.score), [91, 91, 91, 91, 91, 91, 89]);
});

test("display keeps five when the leading scores differ", () => {
  const clubs = [98, 96, 95, 94, 93, 92].map((score) => ({ score }));
  assert.deepEqual(selectForDisplay(clubs), clubs.slice(0, 5));
  assert.deepEqual(selectForDisplay([]), []);
  assert.deepEqual(selectForDisplay(clubs.slice(0, 3)), clubs.slice(0, 3));
});

test("budget ceiling rounds up to the next 25 dollars and handles missing prices", () => {
  assert.equal(maxMsrp([{ msrp: 649 }, { msrp: null }, { msrp: Infinity }], 500), 650);
  assert.equal(maxMsrp([], 500), 500);
});
