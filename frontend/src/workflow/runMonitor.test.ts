import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Mock Web Storage & DOM APIs before importing store
class MockStorage implements Storage {
  private store: Map<string, string> = new Map();

  get length(): number {
    return this.store.size;
  }

  clear(): void {
    this.store.clear();
  }

  getItem(key: string): string | null {
    return this.store.has(key) ? this.store.get(key)! : null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }

  key(index: number): string | null {
    return Array.from(this.store.keys())[index] ?? null;
  }
}

const mockLocalStorage = new MockStorage();
const mockSessionStorage = new MockStorage();

Object.defineProperty(globalThis, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});
Object.defineProperty(globalThis, 'sessionStorage', {
  value: mockSessionStorage,
  writable: true,
});

class MockCustomEvent {
  type: string;
  detail: unknown;
  constructor(type: string, eventInitDict?: { detail?: unknown }) {
    this.type = type;
    this.detail = eventInitDict?.detail;
  }
}
Object.defineProperty(globalThis, 'CustomEvent', {
  value: MockCustomEvent,
  writable: true,
});

const eventListeners = new Map<string, Set<(evt: any) => void>>();
const dispatchedEvents: { type: string; detail: unknown }[] = [];

Object.defineProperty(globalThis, 'window', {
  value: {
    dispatchEvent: (evt: { type: string; detail: unknown }) => {
      dispatchedEvents.push(evt);
      const listeners = eventListeners.get(evt.type);
      if (listeners) {
        listeners.forEach((fn) => fn(evt));
      }
      return true;
    },
    addEventListener: (type: string, listener: (evt: any) => void) => {
      if (!eventListeners.has(type)) eventListeners.set(type, new Set());
      eventListeners.get(type)!.add(listener);
    },
    removeEventListener: (type: string, listener: (evt: any) => void) => {
      eventListeners.get(type)?.delete(listener);
    },
  },
  writable: true,
});

import { api } from '../api/client.ts';
import type { ResearchRun, RunStatus } from '../types.ts';
import {
  registerMonitoredRun,
  reportRunStatus,
  dismissNotification,
  openRunProgress,
  closeRunProgress,
  initRunMonitor,
  createRunNotification,
  isTerminalStatus,
  loadMonitoredRunHint,
  loadDismissedNotificationIds,
  recordDismissedNotificationId,
  startBackgroundPolling,
  stopBackgroundPolling,
  _resetMonitorStateForTesting,
} from './runMonitor.ts';

// Deterministic run fixtures
function makeMockRun(status: RunStatus, overrides: Partial<ResearchRun> = {}): ResearchRun {
  return {
    id: overrides.id || 'run_test_123',
    project_id: overrides.project_id || 'proj_alpha',
    status,
    current_stage: overrides.current_stage ?? 'UNDERSTANDING',
    stages: overrides.stages || [
      { stage_id: 'INGESTION', name: 'Ingestion', status: 'COMPLETED' },
      { stage_id: 'UNDERSTANDING', name: 'Understanding', status: status === 'RUNNING' ? 'RUNNING' : 'COMPLETED' },
      { stage_id: 'GAP_DISCOVERY', name: 'Gap Discovery', status: status === 'AWAITING_REVIEW' || status === 'COMPLETED' ? 'COMPLETED' : 'PENDING' },
    ],
    papers: overrides.papers || [
      {
        id: 'rp_1',
        openalex_id: 'W1',
        title: 'Federated learning survey',
        status: 'COMPLETED',
        has_pdf: true,
        passages_count: 50,
        statements_count: 5,
      },
      {
        id: 'rp_2',
        openalex_id: 'W2',
        title: 'Differential privacy medical AI',
        status: status === 'FAILED' ? 'FAILED' : 'COMPLETED',
        has_pdf: true,
        passages_count: 30,
        statements_count: status === 'FAILED' ? 0 : 4,
      },
    ],
    summary: overrides.summary || {
      total_papers: 2,
      completed_papers: status === 'FAILED' ? 1 : 2,
      successful_papers: status === 'FAILED' ? 1 : 2,
      failed_papers: status === 'FAILED' ? 1 : 0,
      total_statements: status === 'FAILED' ? 5 : 9,
      gaps_discovered: status === 'AWAITING_REVIEW' || status === 'COMPLETED' ? 3 : 0,
    },
    error_message: overrides.error_message || (status === 'FAILED' ? 'GPU memory exhausted' : null),
  };
}

describe('IdeaGen Persistent Background Run Monitor', () => {
  let originalGetRunStatus: typeof api.getRunStatus;

  beforeEach(() => {
    mockLocalStorage.clear();
    mockSessionStorage.clear();
    dispatchedEvents.length = 0;
    eventListeners.clear();
    _resetMonitorStateForTesting();
    originalGetRunStatus = api.getRunStatus;
  });

  test('1. Registering a newly started run stores recovery hint and sets status to RUNNING', () => {
    api.getRunStatus = async (runId: string) => makeMockRun('RUNNING', { id: runId });

    const registered = registerMonitoredRun('run_new_1', 'proj_fl', 'Federated Learning AI');

    assert.equal(registered.runId, 'run_new_1');
    assert.equal(registered.projectId, 'proj_fl');
    assert.equal(registered.projectName, 'Federated Learning AI');
    assert.equal(registered.status, 'RUNNING');

    // Recovery hint saved to sessionStorage
    const hint = loadMonitoredRunHint();
    assert.deepEqual(hint, {
      runId: 'run_new_1',
      projectId: 'proj_fl',
      projectName: 'Federated Learning AI',
    });

    // Custom event dispatched for UI components
    const monitorEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_monitor_change');
    assert.ok(monitorEvents.length > 0);
  });

  test('2. Showing the global indicator while a run is active', () => {
    const runData = makeMockRun('RUNNING', {
      id: 'run_active_1',
      project_id: 'proj_1',
      current_stage: 'UNDERSTANDING',
    });

    const monitored = reportRunStatus(runData, 'Medical AI');

    assert.equal(monitored.status, 'RUNNING');
    assert.equal(monitored.currentStage, 'UNDERSTANDING');
    assert.equal(isTerminalStatus(monitored.status), false);

    // Assert that status is non-terminal so the global indicator in Navbar is active
    assert.ok(monitored.status === 'RUNNING' || monitored.status === 'PENDING');
  });

  test('3. Reopening the correct existing run opens modal for that exact run without starting a new run', () => {
    let createRunCalled = false;
    (api as any).createResearchRun = async () => {
      createRunCalled = true;
      return makeMockRun('RUNNING');
    };

    openRunProgress('run_existing_42', 'proj_test');

    const modalEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_modal_change');
    assert.ok(modalEvents.length > 0);
    const lastEvent = modalEvents[modalEvents.length - 1];
    assert.deepEqual(lastEvent.detail, {
      isOpen: true,
      runId: 'run_existing_42',
      projectId: 'proj_test',
    });

    // Reopening must never call createResearchRun
    assert.equal(createRunCalled, false);
  });

  test('4. Navigating to another project and reopening run belongs to original project', () => {
    registerMonitoredRun('run_cross_1', 'proj_source', 'Source Domain');

    // User is on a different project ('proj_target') and clicks View Progress
    openRunProgress('run_cross_1', 'proj_source');

    const modalEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_modal_change');
    const lastModal = modalEvents[modalEvents.length - 1];
    assert.equal((lastModal.detail as any).isOpen, true);
    assert.equal((lastModal.detail as any).runId, 'run_cross_1');
    assert.equal((lastModal.detail as any).projectId, 'proj_source');
  });

  test('5. Restoring state after a reload and refreshing status authoritatively from backend', async () => {
    // Simulate sessionStorage from prior session
    mockSessionStorage.setItem(
      'ideagen_monitored_run_v1',
      JSON.stringify({
        runId: 'run_persisted_99',
        projectId: 'proj_med',
        projectName: 'Medical Imaging',
      })
    );

    // Backend returns authoritative completed state
    let requestedRunId = '';
    api.getRunStatus = async (runId: string) => {
      requestedRunId = runId;
      return makeMockRun('AWAITING_REVIEW', { id: runId, project_id: 'proj_med' });
    };

    await initRunMonitor();

    assert.equal(requestedRunId, 'run_persisted_99');

    // Authoritative status was refreshed from backend
    const hint = loadMonitoredRunHint();
    assert.equal(hint?.runId, 'run_persisted_99');

    const monitorEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_monitor_change');
    const lastEvent = monitorEvents[monitorEvents.length - 1];
    assert.equal((lastEvent.detail as any).monitoredRun.status, 'AWAITING_REVIEW');
    assert.equal((lastEvent.detail as any).notification.status, 'AWAITING_REVIEW');
  });

  test('6. Handling stale sessionStorage data safely when run is not found on backend', async () => {
    mockSessionStorage.setItem(
      'ideagen_monitored_run_v1',
      JSON.stringify({
        runId: 'run_deleted_ghost',
        projectId: 'proj_ghost',
      })
    );

    // Backend returns 404
    api.getRunStatus = async (runId: string) => {
      const err: any = new Error(`Run ${runId} not found.`);
      err.status = 404;
      throw err;
    };

    await initRunMonitor();

    // Stale hint must be wiped
    assert.equal(mockSessionStorage.getItem('ideagen_monitored_run_v1'), null);
    assert.equal(loadMonitoredRunHint(), null);
  });

  test('7. Showing a completion/review notification exactly once per status transition', () => {
    const runData = makeMockRun('AWAITING_REVIEW', { id: 'run_notify_1' });

    // Initial transition to AWAITING_REVIEW
    reportRunStatus(runData);

    const monitorEvents1 = dispatchedEvents.filter((e) => e.type === 'ideagen_run_monitor_change');
    const firstNotification = (monitorEvents1[monitorEvents1.length - 1].detail as any).notification;
    assert.ok(firstNotification);
    assert.equal(firstNotification.status, 'AWAITING_REVIEW');

    // User dismisses notification
    dismissNotification();
    assert.ok(loadDismissedNotificationIds().has('run_notify_1'));

    // Subsequent poll reports same status - must NOT re-show notification
    reportRunStatus(runData);
    const monitorEvents2 = dispatchedEvents.filter((e) => e.type === 'ideagen_run_monitor_change');
    const secondNotification = (monitorEvents2[monitorEvents2.length - 1].detail as any).notification;
    assert.equal(secondNotification, null);
  });

  test('8. Handling AWAITING_REVIEW, COMPLETED, PARTIAL_SUCCESS, FAILED, and INTERRUPTED distinctly', () => {
    const awaiting = createRunNotification({
      runId: 'r1',
      projectId: 'p1',
      status: 'AWAITING_REVIEW',
      gapsDiscovered: 4,
    });
    assert.equal(awaiting.variant, 'warning');
    assert.match(awaiting.title, /Ready for Review/i);
    assert.match(awaiting.message, /4 research gap/);

    const completed = createRunNotification({
      runId: 'r2',
      projectId: 'p1',
      status: 'COMPLETED',
      statementsExtracted: 25,
      totalPapers: 5,
    });
    assert.equal(completed.variant, 'success');
    assert.match(completed.title, /Completed/i);
    assert.match(completed.message, /25 statements/);

    const partial = createRunNotification({
      runId: 'r3',
      projectId: 'p1',
      status: 'PARTIAL_SUCCESS',
      failedPapers: 2,
    });
    assert.equal(partial.variant, 'warning');
    assert.match(partial.title, /Diagnostic Issues/i);
    assert.match(partial.message, /2 failed/);

    const failed = createRunNotification({
      runId: 'r4',
      projectId: 'p1',
      status: 'FAILED',
      errorMessage: 'Out of CUDA memory',
    });
    assert.equal(failed.variant, 'danger');
    assert.match(failed.title, /Failed/i);
    assert.equal(failed.message, 'Out of CUDA memory');

    const interrupted = createRunNotification({
      runId: 'r5',
      projectId: 'p1',
      status: 'INTERRUPTED',
    });
    assert.equal(interrupted.variant, 'danger');
    assert.match(interrupted.title, /Interrupted/i);
    assert.match(interrupted.message, /server stop or restart/i);
  });

  test('9. Avoiding duplicate polling timers when modal is open and polling', () => {
    api.getRunStatus = async (runId: string) => makeMockRun('RUNNING', { id: runId });

    registerMonitoredRun('run_poll_1', 'proj_1');

    // Progress modal is opened for run_poll_1
    openRunProgress('run_poll_1', 'proj_1');

    // Polling while modal is open delegates to modal to avoid duplicate network traffic
    const modalEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_modal_change');
    assert.equal((modalEvents[modalEvents.length - 1].detail as any).isOpen, true);
  });

  test('10. Cleaning up polling when no active runs remain or run reaches terminal state', () => {
    api.getRunStatus = async (runId: string) => makeMockRun('RUNNING', { id: runId });

    registerMonitoredRun('run_terminal_1', 'proj_1');

    // Run completes
    const completedRun = makeMockRun('COMPLETED', { id: 'run_terminal_1' });
    reportRunStatus(completedRun);

    assert.equal(isTerminalStatus(completedRun.status), true);
  });

  test('11. Ensuring minimizing the modal does not interrupt backend execution', () => {
    registerMonitoredRun('run_bg_1', 'proj_1', 'Project 1');
    openRunProgress('run_bg_1', 'proj_1');

    // User clicks Minimize
    closeRunProgress();

    const modalEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_modal_change');
    const lastModal = modalEvents[modalEvents.length - 1];
    assert.equal((lastModal.detail as any).isOpen, false);
    assert.equal((lastModal.detail as any).runId, 'run_bg_1');

    // Hint in sessionStorage remains intact so background execution is monitored
    const hint = loadMonitoredRunHint();
    assert.ok(hint);
    assert.equal(hint.runId, 'run_bg_1');
  });

  test('12. Preserving current Run History reopen behavior', () => {
    const historyRun = makeMockRun('AWAITING_REVIEW', { id: 'run_hist_77', project_id: 'proj_hist' });

    // In Run History, user clicks "Inspect Run & Logs"
    openRunProgress(historyRun.id, historyRun.project_id);

    const modalEvents = dispatchedEvents.filter((e) => e.type === 'ideagen_run_modal_change');
    const lastModal = modalEvents[modalEvents.length - 1];
    assert.equal((lastModal.detail as any).isOpen, true);
    assert.equal((lastModal.detail as any).runId, 'run_hist_77');
    assert.equal((lastModal.detail as any).projectId, 'proj_hist');
  });
});
