# FIRE Forecast Engine

**One-line pitch:** Simulate your path to financial independence with Monte Carlo projections.

## Problem & Solution

**Problem:** You have a set-and-forget ETF strategy but no way to visualize how different life scenarios (salary bumps, expense changes, market downturns) affect your FIRE timeline.

**Solution:** A Python CLI tool that runs thousands of randomized simulations against your financial profile and shows the probability of reaching financial independence at various ages.

## Target Audience

Personal use — an investor with a simple ETF strategy who wants data-driven confidence in their FIRE plan.

## Core Features (MVP)

1. **Financial Profile** — Define income, expenses, savings rate, current portfolio
2. **Investment Strategies** — Model different approaches (aggressive, balanced, conservative ETF allocations)
3. **Monte Carlo Engine** — Run 1,000+ simulations with randomized market returns
4. **FIRE Analysis** — Calculate probability of retirement at different ages/targets
5. **CLI Reports** — Clear summary with key percentiles (10th, 50th, 90th)

## OOP Concepts & Design Patterns

| Week | Pattern | Where It Shows Up |
|------|---------|-------------------|
| 1 | Classes, Encapsulation, Composition | Portfolio has Assets, Profile has Portfolio |
| 2 | Strategy Pattern | Swappable investment strategies |
| 2 | Template Method | Simulation lifecycle (setup → run → aggregate) |
| 3 | Inheritance & Polymorphism | Different asset types, different market models |
| 4 | Factory Pattern | Creating scenarios from config files |

## Tech Stack

- **Python 3.12+** — core language
- **dataclasses** — bridge from functional thinking to OOP
- **matplotlib** — visualize projection fans
- **pytest** — test-driven where it makes sense
- **YAML/JSON** — scenario configuration

## Learning Goals

- Understand when and why OOP is the right tool (vs functional)
- Implement Strategy and Template Method patterns from scratch
- Practice composition over inheritance
- Build a project with proper domain modeling

## Success Metrics

- Can model your actual financial situation
- Produces projections you trust enough to reference
- You can explain Strategy and Template Method patterns to someone else
