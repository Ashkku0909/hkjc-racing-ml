-- 1. Races Table
CREATE TABLE races (
    race_id SERIAL PRIMARY KEY,
    track VARCHAR(50) NOT NULL,          -- e.g., 'Happy Valley', 'Sha Tin'
    race_date DATE NOT NULL,
    race_class VARCHAR(20),              -- e.g., 'Class 1', 'Group 1'
    distance INTEGER NOT NULL,           -- Stored in meters, e.g., 1200, 1650
    place_dividends TEXT,                -- PLA pool dividend line(s)
    quinella_dividend NUMERIC(6, 2),     -- QIN dividend
    qpl_dividends TEXT,                  -- QPL pair dividends
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(track, race_date, distance)   -- Prevents duplicate race entries
);

-- 2. Horses Table
CREATE TABLE horses (
    horse_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,   -- HKJC horse names are unique
    origin VARCHAR(50),                  -- e.g., 'AUS', 'IRE', 'NZ'
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Race Results Table
CREATE TABLE race_results (
    result_id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    horse_id INTEGER NOT NULL REFERENCES horses(horse_id) ON DELETE CASCADE,
    finish_position VARCHAR(10),         -- VARCHAR to handle '1', 'DNF', 'SCR', '1 DH'
    finishing_time NUMERIC(6, 2),        -- Total seconds (e.g., 82.50)
    jockey VARCHAR(100) NOT NULL,
    trainer VARCHAR(100) NOT NULL,
    weight_carried NUMERIC(5, 2),        -- e.g., 133.0
    barrier_draw INTEGER,                -- e.g., 1 to 14
    win_odds NUMERIC(6, 2),              -- e.g., 4.50
    sec1_time NUMERIC(6, 2),             -- Sectional times
    sec2_time NUMERIC(6, 2),
    sec3_time NUMERIC(6, 2),
    sec4_time NUMERIC(6, 2),
    sec5_time NUMERIC(6, 2),
    sec6_time NUMERIC(6, 2),
    horse_rating INTEGER,                -- Official HKJC rating (NULL for debutants)
    rating_change INTEGER,               -- Rating delta vs previous start
    gear VARCHAR(50),                    -- Equipment, e.g. 'B', 'TT', 'V', 'B/TT', '--'
    jockey_allowance INTEGER,            -- Apprentice claim, e.g. -10, -5, -2, 0
    incident_report TEXT,                -- Stewards' racing incident paragraph(s)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(race_id, horse_id)            -- A horse can only have one result per race
);

-- Indexes for faster querying during model training
CREATE INDEX idx_race_date ON races(race_date);
CREATE INDEX idx_horse_name ON horses(name);
