export const USED_MAX_YEAR = new Date().getFullYear() - 4;

export function selectForDisplay(clubs, defaultLimit = 5) {
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

export function filterByCondition(clubs, condition) {
  if (condition !== "used") {
    return clubs;
  }
  return clubs.filter(
    (club) => Number.isInteger(club.year) && club.year <= USED_MAX_YEAR,
  );
}

export function displayYears(club, condition) {
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

export function filterByBudget(clubs, budget) {
  return clubs
    .filter((club) => typeof club.msrp === "number" && club.msrp <= budget)
    .sort((a, b) => b.score - a.score);
}

export function maxMsrp(clubs, fallback) {
  const prices = clubs
    .map((club) => club.msrp)
    .filter((price) => typeof price === "number" && Number.isFinite(price));
  if (!prices.length) {
    return fallback;
  }
  return Math.ceil(Math.max(...prices) / 25) * 25;
}
