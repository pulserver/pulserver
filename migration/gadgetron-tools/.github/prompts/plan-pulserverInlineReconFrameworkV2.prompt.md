## Plan: PulServer Inline Recon Framework V2

Build a minimal-diff reconstruction stack without modifying legacy third-party servers in place: create a small new Python MRD server package that stays close to the reference gadgetron Python server shape, add only required fork features (autosave/logging/TOML + plugin loading), and apply small targeted changes in GEToISMRMRD and GadgetronClient for Docker startup, timeout safety, and metadata enrichment. The design prioritizes never freezing scanner acquisition, never losing raw data, and eventual reconstruction completion.

**Steps**
1. Global quality gate (applies to every phase): no feature is considered complete without unit tests for all new/changed logic and passing test runs in CI.
2. Global scope gate (applies to every phase): prefer smallest possible patch on top of reference code; avoid new layers, new frameworks, and broad refactors unless a blocker is proven.
3. Phase 0: Freeze boundaries and compatibility targets.
4. Keep wire protocol and message handling behavior aligned with `3p/gadgetron-python-ismrmrd-server` and `3p/python-ismrmrd-server` so the current C++ emitter continues to work with minimal churn.
5. Define open-source/proprietary boundary doc: open modules must only consume generic MRD/Pulseq metadata and must not include GE-confidential internals from `3p/gadgetron-emitter/src/Orchestra/GadgetronEmitter/Application/*`.
6. Phase 1: Build a minimal new server package by copying upstream layout and applying focused deltas only. *depends on 1-5*
7. Create new package with file structure matching `main.py`, `server.py`, `connection.py`, `constants.py`; keep function names and control flow close to upstream for easy diff/review.
8. Add only required features from fork: autosave incoming data, structured logging option, TOML/JSON config-text support, and robust close/error handling.
9. Keep hardcoded recon dispatch removed, but do not redesign transport internals.
10. Phase 2: Plugin support with lean ABC contract plus lightweight registry/discovery. *depends on 6-9*
11. Implement one simple ABC contract (for example `process(connection, config, metadata)` on a plugin class).
12. Add lightweight discovery: support explicit module/class loading plus optional scan of a configured plugins directory, with a small registry map (name -> module/class) built at startup.
13. Keep registry scope intentionally small in v1: no plugin dependency graph, no hot-reload, no remote catalogs.
14. Add compatibility wrapper for legacy function-style modules so existing scripts can run during migration.
15. Ship one built-in reference plugin equivalent to `simplefft`.
16. Add one educational reference plugin patterned after the Gadgetron course buffered Python example (SigPy path) to complement GRE reconstruction examples.
17. Phase 3: Non-blocking and overlap-safe behavior with minimal queue semantics. *depends on 6-16*
18. Persist every incoming scan to per-scan MRD file first (durable write), then run recon from that file.
19. Start with a simple FIFO disk-backed pending list and one recon worker process/thread per server instance (configurable), instead of a complex scheduler.
20. If a new scan arrives while recon is running, accept and persist it, enqueue it, and return quickly from ingest path to avoid scanner blocking.
21. Add simple restart recovery: on startup, scan spool folder and continue pending jobs.
22. Phase 4: Minimal C++ open-source changes (GEToISMRMRD + GadgetronClient) with private-app integration kept separate. *depends on 1-5; can start in parallel with Phase 2*
23. In open source, add only the required timeout/connection helpers in `GadgetronClient` and metadata override hooks in `GEToISMRMRD`.
24. Keep `AcquisitionSection.cpp` as private workflow reference only; do not copy private code into public repo. Any private integration remains in private repository.
25. Preserve current config entry flow (`rdb_hdr_user0` to `pulseq*.entry`) at workflow level and add minimal field for plugin module/class name through public interfaces.
26. Phase 5: Minimal metadata enrichment in converter path. *depends on 22-25*
27. Add tiny Pulseq metadata parser for only required fields (matrix/FOV/readout center/trajectory hints), with no full parser framework.
28. Inject only necessary overrides into XML/acquisition headers at existing extension points in `GERawConverter.cpp`; private caller wiring remains private.
29. Phase 6: Validation and rollout. *depends on all prior phases*
30. Add unit tests first for each new behavior (server, plugin loading/registry discovery, queue/recovery, GadgetronClient timeout logic, metadata parser).
31. Add focused integration tests only for critical paths (single scan, back-to-back scans, delayed recon, restart recovery).
32. Run simulation rollout before scanner rollout.

**Relevant files**
- `/home/mcencini/pulserver/3p/gadgetron-python-ismrmrd-server/server.py` — clean baseline for accept/serve/handle flow and multiprocess behavior to mirror.
- `/home/mcencini/pulserver/3p/gadgetron-python-ismrmrd-server/connection.py` — baseline message framing and handlers.
- `/home/mcencini/pulserver/3p/python-ismrmrd-server/server.py` — reference for dynamic module loading and additional config text override behavior.
- `/home/mcencini/pulserver/3p/python-ismrmrd-server/connection.py` — reference autosave implementation and message extensions.
- `/home/mcencini/pulserver/3p/gadgetron-emitter/src/Orchestra/GadgetronEmitter/Application/AcquisitionSection.cpp` — private workflow reference only for behavior/context; do not copy into public repo.
- `/home/mcencini/pulserver/3p/gadgetron-emitter/src/Orchestra/GadgetronClient/gadgetron_ismrmrd_client.cpp` — public open-source target for minimal timeout/connection helper changes.
- `/home/mcencini/pulserver/3p/gadgetron-emitter/src/Orchestra/GEToIsmrmrd/GERawConverter.cpp` — public open-source target for minimal XML/user-parameter override hooks.
- `/home/mcencini/pulserver/csrc/pulseqlib_methods.h` — existing metadata/trajectory getter baseline for future alternative (if tiny parser proves insufficient).
- `/home/mcencini/pulserver/3p/GadgetronOnlineClass/Courses/Day1/Lecture2/demo/config.xml` — practical Gadgetron chain assembly example for a minimal GRE+recon demonstration.
- `/home/mcencini/pulserver/3p/GadgetronOnlineClass/Courses/Day1/Lecture3/Part2/correction/my_first_buffered_data_python_gadget.py` — buffered Python recon examples including `SigPyBufferedDataPythonGadget`.
- `/home/mcencini/pulserver/3p/GadgetronOnlineClass/Courses/Day1/Lecture3/Part2/correction/external_python_buffer_tutorial.xml` — XML execution wiring for buffered Python gadgets.

**Verification**
1. Unit-test completeness gate: every new/modified module must include unit tests in the same PR, and CI must fail on missing tests or failing tests.
2. Server protocol tests: validate all supported message IDs and framing against fixture streams generated by current emitter client.
3. Plugin contract tests: reject invalid plugins, load valid ABC plugins, and execute adapter-wrapped legacy function plugins.
4. Queue durability tests: kill server mid-recon, restart, verify pending jobs continue and complete without data loss.
5. Throughput tests: run back-to-back scan ingest while slowing recon workers; assert scanner-side stream close stays within timeout budget.
6. Disk-pressure tests: simulate low disk; verify degraded mode and safe refusal policy are explicit and logged.
7. Emitter tests: docker preflight/start success and failure branches, connect timeout handling, and non-blocking scan-end logic.
8. Metadata tests: verify generated acquisition headers/XML include expected matrix/FOV/readout-center values from sample Pulseq files.
9. End-to-end system test: emitter -> containerized server -> plugin recon -> delayed image output; verify eventual completion across sequential scans.

**Decisions**
- New recon server package will be created, but implementation must remain structurally close to `3p/gadgetron-python-ismrmrd-server` with minimal file/function deltas.
- Plugin model remains ABC-based, with lightweight registry/discovery in v1 (explicit module/class loading plus optional configured plugins-folder scan).
- Scanner overlap handling is minimal FIFO disk-backed queueing with durable ingest first, not a complex scheduler.
- Docker lifecycle control starts in C++ emitter client before server connection, using bounded timeout only.
- Sequence metadata enrichment first uses a tiny dedicated parser for required fields only; avoid broad pulse sequence framework work.
- Open-source outputs must exclude proprietary GE internals.

**Further Considerations**
1. If explicit ABC loading adds too much complexity, temporary function-wrapper fallback is acceptable as long as the ABC path is still test-covered.
2. If tiny parser grows beyond minimal required fields, pivot to exposing a small pulseqlib getter instead of building a larger custom parser.
3. Keep first rollout with one recon worker and one queue path; scale only after measured need.
