-- SignalScope — Supabase RLS Policy Setup
-- Run this in the Supabase SQL editor (Project → SQL Editor → New Query)
-- This allows the anon key (publishable key) to insert/read analysis data.
--
-- OPTION A: Disable RLS entirely (fastest for hackathon/internal testing)
-- Uncomment the lines below to disable RLS:
--
-- ALTER TABLE analyses      DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE probe_results DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE provenance    DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE feedback      DISABLE ROW LEVEL SECURITY;
--
-- OPTION B: Keep RLS but add permissive policies (recommended)
-- This allows the anon key to insert and read all rows.

-- analyses
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_insert_analyses" ON analyses;
DROP POLICY IF EXISTS "anon_select_analyses" ON analyses;
CREATE POLICY "anon_insert_analyses" ON analyses FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_analyses" ON analyses FOR SELECT TO anon USING (true);

-- probe_results
ALTER TABLE probe_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_insert_probes" ON probe_results;
DROP POLICY IF EXISTS "anon_select_probes" ON probe_results;
CREATE POLICY "anon_insert_probes" ON probe_results FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_probes" ON probe_results FOR SELECT TO anon USING (true);

-- provenance
ALTER TABLE provenance ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_insert_provenance" ON provenance;
DROP POLICY IF EXISTS "anon_select_provenance" ON provenance;
CREATE POLICY "anon_insert_provenance" ON provenance FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_provenance" ON provenance FOR SELECT TO anon USING (true);

-- feedback
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_insert_feedback" ON feedback;
DROP POLICY IF EXISTS "anon_select_feedback" ON feedback;
CREATE POLICY "anon_insert_feedback" ON feedback FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_select_feedback" ON feedback FOR SELECT TO anon USING (true);

-- Verify: list all RLS policies
SELECT schemaname, tablename, policyname, cmd, roles
FROM pg_policies
WHERE tablename IN ('analyses','probe_results','provenance','feedback')
ORDER BY tablename, policyname;
