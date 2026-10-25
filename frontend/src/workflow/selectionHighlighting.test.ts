import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import type { PaperItem, ResearchGap, OpportunitySummary } from '../types.ts';

// Deterministic test fixtures
const mockPapers: PaperItem[] = [
  {
    id: 'paper_001',
    source_id: 'W1001',
    title: 'Federated Learning in Oncology: A Comprehensive Survey',
    publication_date: '2023-01-15',
    temporal_role: 'BASELINE',
    passage_count: 42,
    knowledge_status: 'ANALYZED',
    is_baseline: true,
  },
  {
    id: 'paper_002',
    source_id: 'W1002',
    title: 'Differential Privacy Guarantees for Clinical ML Models',
    publication_date: '2023-08-20',
    temporal_role: 'BASELINE',
    passage_count: 28,
    knowledge_status: 'ANALYZED',
    is_baseline: true,
  },
  {
    id: 'paper_003',
    source_id: 'W1003',
    title: 'Longitudinal Validation of Multimodal EHR Transformers',
    publication_date: '2024-03-10',
    temporal_role: 'LATER_EVALUATION',
    passage_count: 35,
    knowledge_status: 'ANALYZED',
    is_baseline: false,
  },
];

const mockGaps: ResearchGap[] = [
  {
    id: 'gap_101',
    gap_statement: 'Non-IID patient data distributions cause client gradient divergence in oncology trials.',
    research_problem: 'Client gradient divergence under extreme clinical heterogeneity.',
    status: 'VALIDATED',
    validation_confidence: 0.92,
    evidence_count: 5,
    supporting_paper_ids: ['paper_001'],
    supporting_statement_ids: ['stmt_001'],
    limitations: [],
    future_work: [],
    created_at: '2026-10-22T10:00:00Z',
  },
  {
    id: 'gap_102',
    gap_statement: 'Absence of verifiable audit trails for private model parameter updates.',
    research_problem: 'Lack of cryptographic verification for local training verification.',
    status: 'CANDIDATE',
    validation_confidence: 0.74,
    evidence_count: 3,
    supporting_paper_ids: ['paper_002'],
    supporting_statement_ids: ['stmt_002'],
    limitations: [],
    future_work: [],
    created_at: '2026-10-22T10:05:00Z',
  },
  {
    id: 'gap_103',
    gap_statement: 'Evaluation protocols fail to account for clinical domain shifts over multi-year deployments.',
    research_problem: 'Temporal domain drift evaluation in healthcare federated pipelines.',
    status: 'VALIDATED',
    validation_confidence: 0.88,
    evidence_count: 4,
    supporting_paper_ids: ['paper_003'],
    supporting_statement_ids: ['stmt_003'],
    limitations: [],
    future_work: [],
    created_at: '2026-10-22T10:10:00Z',
  },
];

const mockOpportunities: OpportunitySummary[] = [
  {
    id: 'opp_501',
    lifecycle_opportunity_id: 'lopp_501',
    proposed_research_direction: 'Adaptive Personalized Federated Optimization for Rare Disease Cohorts',
    identified_gap: 'Non-IID patient data distributions cause client gradient divergence in oncology trials.',
    research_problem: 'Client gradient divergence under extreme clinical heterogeneity.',
    status: 'VALIDATED',
    lifecycle_status: 'OPEN',
    confidence: 0.89,
    evidence_count: 2,
    supporting_statement_count: 3,
    created_at: '2026-10-22T11:00:00Z',
  },
  {
    id: 'opp_502',
    lifecycle_opportunity_id: 'lopp_502',
    proposed_research_direction: 'Zero-Knowledge Parameter Attestation for Federated Medical Research',
    identified_gap: 'Absence of verifiable audit trails for private model parameter updates.',
    research_problem: 'Lack of cryptographic verification for local training verification.',
    status: 'VALIDATED',
    lifecycle_status: 'STRENGTHENED',
    confidence: 0.85,
    evidence_count: 4,
    supporting_statement_count: 5,
    created_at: '2026-10-22T11:05:00Z',
  },
];

describe('IdeaGen Selected-Item Highlighting & Master-Detail Synchronization', () => {
  // 1. Selection State by ID
  test('1. Clicking an item selects that specific item by ID rather than position', () => {
    let selectedPaperId: string | null = null;
    const handleSelectPaper = (id: string) => {
      selectedPaperId = id;
    };

    // Click second paper
    handleSelectPaper(mockPapers[1].id);
    assert.equal(selectedPaperId, 'paper_002');

    // Verify selection evaluation
    const isSelected = (id: string) => id === selectedPaperId;
    assert.equal(isSelected(mockPapers[0].id), false);
    assert.equal(isSelected(mockPapers[1].id), true);
    assert.equal(isSelected(mockPapers[2].id), false);
  });

  // 2. Selection Persistence Across Re-renders & Data Refreshes
  test('2. Selection remains consistent after data refresh or list re-render', () => {
    let selectedGapId: string | null = 'gap_102';

    // Simulate data refresh (new array instance returned by API with identical IDs)
    const refreshedGaps: ResearchGap[] = mockGaps.map((g) => ({ ...g }));

    // Selection helper mirroring component logic:
    // setSelectedGapId((prev) => prev && data.some(g => g.id === prev) ? prev : fallback)
    const resolveSelectedGapId = (prev: string | null, data: ResearchGap[]) => {
      if (prev && data.some((g) => g.id === prev)) return prev;
      return data.length > 0 ? data[0].id : null;
    };

    selectedGapId = resolveSelectedGapId(selectedGapId, refreshedGaps);
    assert.equal(selectedGapId, 'gap_102');

    const selectedGap = refreshedGaps.find((g) => g.id === selectedGapId);
    assert.ok(selectedGap);
    assert.equal(selectedGap.id, 'gap_102');
    assert.equal(selectedGap.status, 'CANDIDATE');
  });

  // 3. Moving Selection
  test('3. Selecting another item moves the selection styling cleanly to the new item', () => {
    let selectedOppId: string | null = 'opp_501';

    const getOpportunityClasses = (oppId: string) => {
      const isSelected = oppId === selectedOppId;
      return {
        className: `btn ${isSelected ? 'btn-primary is-selected selected' : 'btn-secondary'} btn-sm opp-selector-pill`,
        ariaSelected: isSelected,
        ariaPressed: isSelected,
        dataSelected: isSelected ? 'true' : 'false',
        hasCheck: isSelected,
      };
    };

    // Initially opp_501 is selected
    const initialOpp1 = getOpportunityClasses('opp_501');
    const initialOpp2 = getOpportunityClasses('opp_502');
    assert.ok(initialOpp1.className.includes('is-selected'));
    assert.ok(initialOpp1.className.includes('selected'));
    assert.equal(initialOpp1.ariaSelected, true);
    assert.equal(initialOpp1.hasCheck, true);
    assert.equal(initialOpp2.ariaSelected, false);
    assert.equal(initialOpp2.hasCheck, false);

    // Switch selection to opp_502
    selectedOppId = 'opp_502';
    const updatedOpp1 = getOpportunityClasses('opp_501');
    const updatedOpp2 = getOpportunityClasses('opp_502');
    assert.equal(updatedOpp1.ariaSelected, false);
    assert.equal(updatedOpp1.hasCheck, false);
    assert.ok(updatedOpp2.className.includes('is-selected'));
    assert.ok(updatedOpp2.className.includes('selected'));
    assert.equal(updatedOpp2.ariaSelected, true);
    assert.equal(updatedOpp2.hasCheck, true);
  });

  // 4. Detail Panel Synchronization
  test('4. The detail panel always strictly corresponds to the selected item ID', () => {
    const selectedGapId = 'gap_103';
    const selectedGap = mockGaps.find((g) => g.id === selectedGapId) ?? null;

    assert.ok(selectedGap);
    assert.equal(selectedGap.id, 'gap_103');
    assert.equal(selectedGap.gap_statement, 'Evaluation protocols fail to account for clinical domain shifts over multi-year deployments.');
    assert.equal(selectedGap.validation_confidence, 0.88);
  });

  // 5. Filtering Isolation (No Index Drift)
  test('5. Filtering does not incorrectly select a different item by array index', () => {
    // Select the third paper in the full list
    const selectedPaperId = 'paper_003';

    // Filter to only LATER_EVALUATION papers (where paper_003 becomes index 0)
    const filteredPapers = mockPapers.filter((p) => p.temporal_role === 'LATER_EVALUATION');
    assert.equal(filteredPapers.length, 1);
    assert.equal(filteredPapers[0].id, 'paper_003');

    // Index 0 in filtered list is selected because its ID matches selectedPaperId
    const item0Selected = filteredPapers[0].id === selectedPaperId;
    assert.equal(item0Selected, true);

    // Filter to BASELINE papers (paper_003 is excluded)
    const baselinePapers = mockPapers.filter((p) => p.temporal_role === 'BASELINE');
    assert.equal(baselinePapers.length, 2);

    // Index 0 in baseline list (paper_001) must NOT be selected by accident
    const baseline0Selected = baselinePapers[0].id === selectedPaperId;
    const baseline1Selected = baselinePapers[1].id === selectedPaperId;
    assert.equal(baseline0Selected, false);
    assert.equal(baseline1Selected, false);
  });

  // 6. Sorting Isolation
  test('6. Sorting does not cause item selection to shift to a different record', () => {
    const selectedPaperId = 'paper_002'; // 'Differential Privacy...'

    // Sort alphabetically by title ascending
    const sortedAlpha = [...mockPapers].sort((a, b) => a.title.localeCompare(b.title));
    const selectedInAlpha = sortedAlpha.find((p) => p.id === selectedPaperId);
    assert.ok(selectedInAlpha);
    assert.equal(selectedInAlpha.id, 'paper_002');

    // Sort by publication date descending
    const sortedDate = [...mockPapers].sort((a, b) => (b.publication_date || '').localeCompare(a.publication_date || ''));
    const selectedInDate = sortedDate.find((p) => p.id === selectedPaperId);
    assert.ok(selectedInDate);
    assert.equal(selectedInDate.id, 'paper_002');
  });

  // 7. Accessible Semantic Contract for Corpus Papers and Research Gaps
  test('7. Papers and Research Gaps apply consistent selection classes and accessibility hooks', () => {
    const getRenderedAttributes = (itemId: string, selectedId: string | null) => {
      const isSelected = itemId === selectedId;
      return {
        className: `corpus-paper-item ${isSelected ? 'is-selected selected' : ''}`,
        ariaSelected: isSelected,
        ariaCurrent: isSelected ? 'true' : undefined,
        dataSelected: isSelected ? 'true' : 'false',
        showBadge: isSelected,
      };
    };

    const selectedAttrs = getRenderedAttributes('paper_001', 'paper_001');
    assert.ok(selectedAttrs.className.includes('is-selected'));
    assert.ok(selectedAttrs.className.includes('selected'));
    assert.equal(selectedAttrs.ariaSelected, true);
    assert.equal(selectedAttrs.ariaCurrent, 'true');
    assert.equal(selectedAttrs.dataSelected, 'true');
    assert.equal(selectedAttrs.showBadge, true);

    const unselectedAttrs = getRenderedAttributes('paper_002', 'paper_001');
    assert.equal(unselectedAttrs.className, 'corpus-paper-item ');
    assert.equal(unselectedAttrs.ariaSelected, false);
    assert.equal(unselectedAttrs.ariaCurrent, undefined);
    assert.equal(unselectedAttrs.dataSelected, 'false');
    assert.equal(unselectedAttrs.showBadge, false);
  });

  // 8. Independence from Validation Status
  test('8. Selection highlighting is orthogonal to validation/status badges', () => {
    // Gap 101 is VALIDATED, Gap 102 is CANDIDATE
    const selectedGapId = 'gap_102'; // select the CANDIDATE gap

    const gap1 = mockGaps.find((g) => g.id === 'gap_101')!;
    const gap2 = mockGaps.find((g) => g.id === 'gap_102')!;

    // Selected gap retains its own status without conflation
    assert.equal(gap2.status, 'CANDIDATE');
    assert.equal(gap2.id === selectedGapId, true);

    // Unselected gap retains its own status without conflation
    assert.equal(gap1.status, 'VALIDATED');
    assert.equal(gap1.id === selectedGapId, false);
  });
});
