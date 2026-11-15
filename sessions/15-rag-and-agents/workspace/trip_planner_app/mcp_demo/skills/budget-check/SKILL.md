---
name: budget-check
description: How to check whether a trip fits the budget and how to report it. Use when the user gives a budget.
---

# Budget check

1. Get the return fare and the price of one bed from `search_trip_guide`. Never guess a price.
2. Call `budget_per_person` with the group budget, the number of people, the nights, the fare and the bed price.
3. Report three numbers: the budget per person, what is left for food and activities, and whether the trip is within budget.
4. If `within_budget` is false, say by how many euros per person the trip is over. Do not recommend it.
5. If the guide has no note for the city, say that the prices are unknown and the budget cannot be checked.
6. Say that the prices are course examples, not real offers.
