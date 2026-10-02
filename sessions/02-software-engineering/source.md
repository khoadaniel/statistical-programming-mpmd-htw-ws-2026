# Sources

Third-party material in this session, with its origin and licence. Keep the attribution when you reuse or share a file.

## Workbooks

| File | Covers | Source | Licence | Downloaded | Changes |
|---|---|---|---|---|---|
| [workbooks/01-classes-and-objects.ipynb](workbooks/01-classes-and-objects.ipynb) | Points, lines and rectangles: objects that contain objects (composition), equivalence and identity, deep copy, polymorphism | [AllenDowney/ThinkPython](https://raw.githubusercontent.com/AllenDowney/ThinkPython/v3/chapters/chap16.ipynb) | [MIT (code) / CC BY-NC-SA 4.0 (text)](https://allendowney.github.io/ThinkPython/) **NC: non-commercial only** | 2026-10-01 | Renamed from `chap16.ipynb`; content unchanged |
| [workbooks/02-inheritance.ipynb](workbooks/02-inheritance.ipynb) | Cards and decks: class attributes, parent and child classes, specialisation | [AllenDowney/ThinkPython](https://raw.githubusercontent.com/AllenDowney/ThinkPython/v3/chapters/chap17.ipynb) | [MIT (code) / CC BY-NC-SA 4.0 (text)](https://allendowney.github.io/ThinkPython/) **NC: non-commercial only** | 2026-10-01 | Renamed from `chap17.ipynb`; content unchanged |

> [!NOTE]
> The *Think Python* notebooks may be shared and adapted for non-commercial teaching with attribution and under the same licence. On first run they download the helper files `thinkpython.py` and `diagram.py` (MIT) from the author's repository.

## Linked, not copied

| Resource | Covers | Licence | Why linked |
|---|---|---|---|
| [Software Carpentry: Version Control with Git](https://swcarpentry.github.io/git-novice/) | Git basics, step by step | CC-BY 4.0 | Lesson website (Markdown episodes), not notebooks |
| [CodeRefinery: Collaborative distributed version control](https://coderefinery.github.io/git-collaborative/) | Pull requests, review, forks | CC-BY 4.0 | Lesson website |
| [Carpentries Incubator: Intermediate Research Software Development](https://carpentries-incubator.github.io/python-intermediate-development/) | Testing, design, CI, review around one project | CC-BY 4.0 | Lesson website |
| [MIT Missing Semester 2026](https://missing.csail.mit.edu/) | Version control, code quality, agentic coding | CC BY-NC-SA 4.0 | Non-commercial licence; linked only |
| [Pro Git, 2nd edition](https://git-scm.com/book/en/v2) | Git reference | CC BY-NC-SA 3.0 | Book |
| [Learn Git Branching](https://learngitbranching.js.org/) | Interactive branching exercises | MIT | Web application |
| [GitHub Skills](https://github.com/skills/review-pull-requests) | Self-checking exercises on pull requests and merge conflicts | MIT | Run on GitHub |

## Own material

| File | Covers | Licence |
|---|---|---|
| [theory/01-oop-errors-and-tests.md](theory/01-oop-errors-and-tests.md) | Composition, inheritance, dataclasses, type hints, exceptions, pytest | CC-BY-4.0, course team |
| [theory/02-web-apis.md](theory/02-web-apis.md) | HTTP, JSON, authentication, pagination, rate limits, httpx, FastAPI | CC-BY-4.0, course team |
| [theory/03-git-github-and-ci.md](theory/03-git-github-and-ci.md) | Git, merge-conflict exercise, pull requests, review, GitHub Actions | CC-BY-4.0, course team |
| [workbooks/03-case-study-open-data-api.ipynb](workbooks/03-case-study-open-data-api.ipynb) | Live CKAN request, pagination, validated parsing, MockTransport test | CC-BY-4.0, course team |
| [workspace/](workspace/README.md) | Project `btitools` (BTI decision records, open-data client, FastAPI app with a small nomenclature extract, text features, tests, CI, solutions) | CC-BY-4.0 (text), MIT (code), course team |

Code examples adapt the verified examples of the earlier course notes. The open-data client queries the CKAN API of GovData (https://www.govdata.de); no data are stored in the repository. The nomenclature extract in `workspace/src/btitools/api.py` (nine headings, shortened English descriptions) comes from the HS 2022 dataset cited below; the example decisions in the tests are invented.

## Citations

- Chacon, S., & Straub, B. (2014). *Pro Git*, 2nd edition. Apress. https://git-scm.com/book/en/v2
- CKAN Association (2026). *CKAN API guide*. https://docs.ckan.org/en/latest/api/
- CodeRefinery (2025). *Collaborative distributed version control*. https://coderefinery.github.io/git-collaborative/
- Downey, A. B. (2024). *Think Python*, 3rd edition. O'Reilly. https://allendowney.github.io/ThinkPython/
- European Commission (2026). *European Binding Tariff Information (EBTI) database* (course data; reuse under Commission Decision 2011/833/EU). https://ec.europa.eu/taxation_customs/dds2/ebti/ebti_consultation.jsp?Lang=en
- Google (2026). *Google Engineering Practices: Code review*. https://google.github.io/eng-practices/review/
- Herndon, T., Ash, M., & Pollin, R. (2014). Does high public debt consistently stifle economic growth? A critique of Reinhart and Rogoff. *Cambridge Journal of Economics*, 38(2), 257–279. https://doi.org/10.1093/cje/bet075
- Lehtosalo, J. et al. (2019). *Our journey to type checking 4 million lines of Python*. Dropbox Tech Blog. https://dropbox.tech/application/our-journey-to-type-checking-4-million-lines-of-python
- MIT CSAIL (2026). *The Missing Semester of Your CS Education*. https://missing.csail.mit.edu/
- Mozilla (2026). *MDN Web Docs: HTTP*. https://developer.mozilla.org/en-US/docs/Web/HTTP
- Open Knowledge Foundation / datasets (2026). *Harmonized System (HS) nomenclature*, HS 2022, ODC-PDDL-1.0. https://github.com/datasets/harmonized-system
- Python Software Foundation (2026). *dataclasses — Data Classes*. https://docs.python.org/3/library/dataclasses.html
- Ramírez, S. (2026). *FastAPI documentation*. https://fastapi.tiangolo.com/
- Smith, E. V. (2017). *PEP 557 – Data Classes*. https://peps.python.org/pep-0557/
- Software Carpentry (2024). *Version Control with Git*. https://swcarpentry.github.io/git-novice/
- The Carpentries Incubator (2024). *Intermediate Research Software Development in Python*. https://carpentries-incubator.github.io/python-intermediate-development/
- BBC News (2020, 5 October). *Covid: Test error should never have happened – Hancock*. https://www.bbc.com/news/uk-54422505
