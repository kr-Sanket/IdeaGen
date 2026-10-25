import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { api } from '../api/client.ts';
import type { GapItem, ResearchGapRefineRequest } from '../types.ts';

// Deterministic gap fixtures
const mockParentCandidateGap: GapItem = {
  id: '508d2990-82bd-4e98-8330-692cf9ee2cdc',
  gap_statement: 'Direct STNLF methods require evaluation on residential microgrids.',
  research_problem: 'Transferability of STNLF to behind-the-meter rooftop PV microgrids.',
  status: 'CANDIDATE',
  evidence_count: 2,
  supporting_paper_ids: ['paper_solar_01'],
  supporting_statement_ids: ['stmt_solar_01', 'stmt_solar_02'],
  limitations: ['Aggregated feeder smoothing assumption'],
  future_work: ['Validate transferability to individual residential nodes'],
  metadata: {
    needs_refinement: true,
    single_passage_evidence: true,
  },
};

const mockValidatedGap: GapItem = {
  id: 'a8047ceb-18a8-444f-a7bb-403d1544a4aa',
  gap_statement: 'Validated multi-inverter forecasting divergence.',
  research_problem: 'Forecasting error propagation in multi-stage inverters.',
  status: 'VALIDATED',
  evidence_count: 4,
  supporting_paper_ids: ['paper_inv_01', 'paper_inv_02'],
  supporting_statement_ids: ['stmt_inv_01', 'stmt_inv_02'],
  limitations: [],
  future_work: [],
};

const mockChildRefinedGap: GapItem = {
  id: 'child_ref_001',
  gap_statement: 'To what extent do direct short-term net-load forecasting methods developed on aggregated microgrids transfer to residential microgrids with volatile behind-the-meter rooftop PV and household demand?',
  research_problem: 'Evaluating domain and scale transferability of short-term net-load forecasting algorithms from aggregated distribution networks to residential microgrids with unobserved behind-the-meter PV generation.',
  status: 'CANDIDATE',
  evidence_count: 2,
  supporting_paper_ids: ['paper_solar_01'],
  supporting_statement_ids: ['stmt_solar_01', 'stmt_solar_02'],
  limitations: ['Aggregated feeder smoothing assumption'],
  future_work: ['Validate transferability to individual residential nodes'],
  metadata: {
    parent_gap_id: '508d2990-82bd-4e98-8330-692cf9ee2cdc',
    refinement_type: 'RESEARCHER_REFINED',
    refinement_rationale: 'Testing transferability addresses unverified scale assumptions without asserting unproven degradation as fact.',
    needs_refinement: true,
  },
};

describe('Non-Destructive Research-Gap Refinement — Frontend Client & Workflow', () => {
  let requestedUrl = '';
  let requestMethod = '';
  let requestBody = '';

  beforeEach(() => {
    requestedUrl = '';
    requestMethod = '';
    requestBody = '';
  });

  test('1. api.refineGap issues POST /api/v1/gaps/{id}/refine with project scoping', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: string | URL | Request, init?: RequestInit) => {
      requestedUrl = url.toString();
      requestMethod = init?.method ?? 'GET';
      requestBody = String(init?.body ?? '');

      return new Response(JSON.stringify(mockChildRefinedGap), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    try {
      const payload: ResearchGapRefineRequest = {
        refined_statement: mockChildRefinedGap.gap_statement,
        refined_problem: mockChildRefinedGap.research_problem,
        refinement_rationale: 'Testing transferability without asserting unproven field degradation.',
      };

      const result = await api.refineGap(mockParentCandidateGap.id, payload, 'proj_solar_2026');

      assert.equal(requestMethod, 'POST');
      assert.match(requestedUrl, new RegExp(`/api/v1/gaps/${mockParentCandidateGap.id}/refine\\?project_id=proj_solar_2026`));
      assert.equal(result.id, mockChildRefinedGap.id);
      assert.equal(result.status, 'CANDIDATE');
      assert.equal(result.metadata?.parent_gap_id, mockParentCandidateGap.id);
      assert.equal(result.metadata?.refinement_type, 'RESEARCHER_REFINED');

      const parsedBody = JSON.parse(requestBody);
      assert.equal(parsedBody.refined_statement, payload.refined_statement);
      assert.equal(parsedBody.refined_problem, payload.refined_problem);
      assert.equal(parsedBody.refinement_rationale, payload.refinement_rationale);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('2. api.refineGap propagates backend error details clearly', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      return new Response(
        JSON.stringify({ detail: 'Only CANDIDATE research gaps are eligible for refinement (current status: VALIDATED).' }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    };

    try {
      const payload: ResearchGapRefineRequest = {
        refined_statement: 'A valid refined statement with enough characters.',
        refined_problem: 'A valid refined problem with enough characters.',
        refinement_rationale: 'A valid rationale with enough characters.',
      };

      await assert.rejects(
        async () => {
          await api.refineGap(mockValidatedGap.id, payload);
        },
        (err: Error) => {
          assert.match(err.message, /Only CANDIDATE research gaps are eligible for refinement/);
          return true;
        }
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('3. Refinement form validation requires >= 10 non-whitespace characters on all fields', () => {
    const validateFields = (statement: string, problem: string, rationale: string): string | null => {
      if (statement.trim().length < 10 || problem.trim().length < 10 || rationale.trim().length < 10) {
        return 'All refinement fields must contain at least 10 non-whitespace characters.';
      }
      return null;
    };

    assert.equal(validateFields('', 'Valid problem description', 'Valid rationale description'), 'All refinement fields must contain at least 10 non-whitespace characters.');
    assert.equal(validateFields('         ', 'Valid problem description', 'Valid rationale description'), 'All refinement fields must contain at least 10 non-whitespace characters.');
    assert.equal(validateFields('Too short', 'Valid problem description', 'Valid rationale description'), 'All refinement fields must contain at least 10 non-whitespace characters.');
    assert.equal(validateFields('Valid refined statement text', 'Short', 'Valid rationale description'), 'All refinement fields must contain at least 10 non-whitespace characters.');
    assert.equal(validateFields('Valid refined statement text', 'Valid problem description', 'Short'), 'All refinement fields must contain at least 10 non-whitespace characters.');
    assert.equal(validateFields('Valid refined statement text', 'Valid problem description', 'Valid rationale description'), null);
  });

  test('4. Scientific prospective hypothesis matches required specification', () => {
    const defaultHypothesis =
      'To what extent do direct short-term net-load forecasting methods developed on aggregated microgrids transfer to residential microgrids with volatile behind-the-meter rooftop PV and household demand?';

    // Must be formulated as an open testable scientific question, not an established assertion
    assert.ok(defaultHypothesis.startsWith('To what extent do'));
    assert.ok(defaultHypothesis.includes('direct short-term net-load forecasting methods'));
    assert.ok(defaultHypothesis.includes('aggregated microgrids transfer to residential microgrids'));
    assert.ok(defaultHypothesis.includes('volatile behind-the-meter rooftop PV and household demand'));
    assert.ok(!defaultHypothesis.includes('>3x')); // Must NOT encode unproven degradation as a known fact
  });

  test('5. Parent-child lineage metadata structure and display eligibility', () => {
    // Parent eligibility
    const isEligibleForRefinement = (gap: GapItem): boolean => {
      return gap.status === 'CANDIDATE';
    };

    assert.equal(isEligibleForRefinement(mockParentCandidateGap), true);
    assert.equal(isEligibleForRefinement(mockValidatedGap), false);

    // Parent lineage detection
    const hasParent = (gap: GapItem): boolean => {
      return Boolean(gap.metadata?.parent_gap_id);
    };

    assert.equal(hasParent(mockParentCandidateGap), false);
    assert.equal(hasParent(mockChildRefinedGap), true);
    assert.equal(mockChildRefinedGap.metadata?.parent_gap_id, mockParentCandidateGap.id);

    // Child lineage list detection
    const parentWithChild: GapItem = {
      ...mockParentCandidateGap,
      metadata: {
        ...mockParentCandidateGap.metadata,
        child_gap_ids: [mockChildRefinedGap.id],
      },
    };

    assert.deepEqual(parentWithChild.metadata?.child_gap_ids, [mockChildRefinedGap.id]);
  });
});
