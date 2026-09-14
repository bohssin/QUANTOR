-- QUANTOR library. Plan §16.
--
-- What this schema is for: making "what did we actually try, and what came of
-- it" a query rather than a memory. Three properties are enforced here rather
-- than left to callers, because each one is a research-integrity claim:
--
--   1. Versions form a TREE (parent_version), not a list. "Go back to the one
--      from Tuesday and branch from there" has to work, and a list cannot
--      express it.
--   2. Nothing is deleted. `archived` hides a strategy from the default view
--      and leaves every run, every rejection and every reason still queryable —
--      the failure taxonomy in §16.4 IS the archive.
--   3. Comparison counts accumulate per FAMILY, never per sweep. The deflated
--      Sharpe ratio (§12) divides by the number of things tried; counting one
--      sweep at a time inflates every result a strategy ever produces.
--
-- Artifacts (equity curves, trade lists, optimization tables, cached bars) are
-- content-addressed on disk and referenced here by digest, so identical results
-- store once and no row ever points at a file that was overwritten in place.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- --- data ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS data_source (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    path             TEXT NOT NULL,
    digest           TEXT NOT NULL,          -- sha256, or "fp:..." fingerprint
    kind             TEXT NOT NULL,          -- tick | bar
    -- Two different timeframes, and conflating them is a silent factor-of-√n
    -- error in every annualized metric. `base_timeframe` is where the bars are
    -- CACHED (M1, so coarser ones are free). `default_timeframe` is what the
    -- source was loaded AS, and what a run uses when the caller does not say.
    -- Defaulting a run to the base instead of the requested timeframe reports
    -- an M1 Sharpe for an M15 strategy: same trades, √15 ≈ 3.9x the number.
    base_timeframe    TEXT NOT NULL,
    default_timeframe TEXT NOT NULL DEFAULT '',
    utc_offset_hours REAL NOT NULL,
    -- Both counts, for the same reason there are two timeframes: showing the
    -- M1 count under an "M15" label is a lie the UI would repeat everywhere.
    default_bars     INTEGER NOT NULL DEFAULT 0,
    rows             INTEGER NOT NULL,       -- source rows (ticks, or bars)
    bars             INTEGER NOT NULL,       -- bars at base_timeframe
    first_ms         INTEGER NOT NULL,
    last_ms          INTEGER NOT NULL,
    tick_size        REAL NOT NULL,
    instrument_json  TEXT NOT NULL,
    quality_json     TEXT NOT NULL,
    bars_sha         TEXT,                   -- artifact holding the bar arrays
    ingest           TEXT NOT NULL DEFAULT 'whole',   -- whole | stream
    created_at       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_data_digest ON data_source (digest);

-- --- strategies ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS strategy (
    id          INTEGER PRIMARY KEY,
    strategy_id TEXT NOT NULL UNIQUE,        -- the name the agent uses
    name        TEXT NOT NULL,
    -- Variants of one idea share a family so their comparison counts add up.
    family      TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    archived    INTEGER NOT NULL DEFAULT 0,
    archived_reason TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_family ON strategy (family);

CREATE TABLE IF NOT EXISTS strategy_version (
    id             INTEGER PRIMARY KEY,
    strategy_pk    INTEGER NOT NULL REFERENCES strategy (id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,
    parent_version INTEGER,                  -- NULL for a root; a tree, not a list
    signal_block   TEXT NOT NULL,
    signal_sha     TEXT NOT NULL,            -- identical code across strategies is visible
    params_json    TEXT NOT NULL DEFAULT '{}',
    description    TEXT NOT NULL DEFAULT '',
    -- Attribution, when a version came from the LuxAlgo Library. Its licence is
    -- free WITH attribution, and the condition attaches per derived work — so
    -- the URL belongs on the version, not in a README someone may not read.
    source_url     TEXT NOT NULL DEFAULT '',
    origin         TEXT NOT NULL DEFAULT 'agent',   -- agent | human | library
    warnings_json  TEXT NOT NULL DEFAULT '[]',
    created_at     REAL NOT NULL,
    UNIQUE (strategy_pk, version)
);

CREATE INDEX IF NOT EXISTS idx_version_strategy ON strategy_version (strategy_pk);

-- --- runs ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS run (
    id            INTEGER PRIMARY KEY,
    version_pk    INTEGER NOT NULL REFERENCES strategy_version (id) ON DELETE CASCADE,
    data_pk       INTEGER NOT NULL REFERENCES data_source (id),
    kind          TEXT NOT NULL,             -- backtest | optimize | validate
    params_json   TEXT NOT NULL DEFAULT '{}',
    timeframe     TEXT NOT NULL,
    modeling_mode TEXT NOT NULL DEFAULT 'bar_close',
    seed          INTEGER,
    status        TEXT NOT NULL,             -- ok | error
    metrics_json  TEXT NOT NULL DEFAULT '{}',
    artifact_sha  TEXT,                      -- equity curve + trade list
    error         TEXT NOT NULL DEFAULT '',
    started_at    REAL NOT NULL,
    finished_at   REAL,
    seconds       REAL
);

CREATE INDEX IF NOT EXISTS idx_run_version ON run (version_pk, id DESC);
CREATE INDEX IF NOT EXISTS idx_run_kind    ON run (kind, id DESC);

CREATE TABLE IF NOT EXISTS optimization (
    run_pk      INTEGER PRIMARY KEY REFERENCES run (id) ON DELETE CASCADE,
    mode        TEXT NOT NULL,               -- grid | random
    objective   TEXT NOT NULL,
    budget_json TEXT NOT NULL DEFAULT '{}',
    comparisons INTEGER NOT NULL,
    results_sha TEXT,                        -- the full result table
    robustness  REAL,                        -- mean(neighbours)/peak, §11
    verdict     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS validation (
    run_pk       INTEGER PRIMARY KEY REFERENCES run (id) ON DELETE CASCADE,
    mode         TEXT NOT NULL,              -- walk_forward | purged_kfold
    folds        INTEGER NOT NULL,
    efficiency   REAL,
    comparisons  INTEGER NOT NULL DEFAULT 0,
    verdict      TEXT NOT NULL DEFAULT '',
    folds_json   TEXT NOT NULL DEFAULT '[]'
);

-- --- provenance ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_session (
    id               INTEGER PRIMARY KEY,
    session_key      TEXT NOT NULL UNIQUE,
    surface          TEXT NOT NULL DEFAULT '',   -- mcp | api | cli
    model            TEXT NOT NULL DEFAULT '',
    runtime          TEXT NOT NULL DEFAULT '',
    mcp_servers_json TEXT NOT NULL DEFAULT '[]',
    transcript_path  TEXT NOT NULL DEFAULT '',
    started_at       REAL NOT NULL
);

-- Every optimization ever run against a family, which is the denominator the
-- deflated Sharpe ratio needs. A per-sweep count is the classic way to make an
-- overfit strategy look significant.
CREATE VIEW IF NOT EXISTS family_comparisons AS
SELECT s.family              AS family,
       COUNT(o.run_pk)       AS sweeps,
       COALESCE(SUM(o.comparisons), 0) AS comparisons
FROM strategy s
JOIN strategy_version v ON v.strategy_pk = s.id
JOIN run r              ON r.version_pk = v.id
JOIN optimization o     ON o.run_pk = r.id
GROUP BY s.family;
