/**
 * Persistent Background Research Run Monitor Store.
 *
 * Provides a shared, reactive controller that tracks in-flight and recently completed
 * research runs across the application, survives page reloads via sessionStorage,
 * coordinates a single non-overlapping polling loop, and emits distinct notifications
 * for terminal states (AWAITING_REVIEW, COMPLETED, PARTIAL_SUCCESS, FAILED, INTERRUPTED).
 */

import { useState, useEffect } from 'react';
import { api } from '../api/client.ts';
import type { ResearchRun, RunStatus } from '../types.ts';

export interface MonitoredRun {
  runId: string;
  projectId: string;
  projectName?: string;
  status: RunStatus;
  currentStage?: string | null;
  stageProgress?: any;
  gapsDiscovered?: number;
  statementsExtracted?: number;
  failedPapers?: number;
  totalPapers?: number;
  errorMessage?: string | null;
  lastUpdated?: string;
}

export interface RunCompletionNotification {
  runId: string;
  projectId: string;
  projectName?: string;
  status: RunStatus;
  title: string;
  message: string;
  variant: 'success' | 'warning' | 'danger' | 'info';
  gapsDiscovered?: number;
  statementsExtracted?: number;
  failedPapers?: number;
  errorMessage?: string | null;
}

export interface ProgressModalState {
  isOpen: boolean;
  runId: string | null;
  projectId: string | null;
}

const STORAGE_KEY_MONITORED_RUN = 'ideagen_monitored_run_v1';
const STORAGE_KEY_DISMISSED_NOTIFICATIONS = 'ideagen_dismissed_run_notifications_v1';
const MONITOR_CHANGE_EVENT = 'ideagen_run_monitor_change';
const MODAL_CHANGE_EVENT = 'ideagen_run_modal_change';

const BACKGROUND_POLL_INTERVAL_MS = 3500;
const MAX_CONSECUTIVE_ERRORS = 5;

// In-memory module state
let currentMonitoredRun: MonitoredRun | null = null;
let currentNotification: RunCompletionNotification | null = null;
let modalState: ProgressModalState = { isOpen: false, runId: null, projectId: null };

let pollTimer: ReturnType<typeof setTimeout> | null = null;
let consecutiveErrors = 0;
let isPollingActive = false;
let isInitialized = false;

export function isTerminalStatus(status: RunStatus): boolean {
  return ['COMPLETED', 'AWAITING_REVIEW', 'PARTIAL_SUCCESS', 'FAILED', 'INTERRUPTED'].includes(status);
}

export function loadDismissedNotificationIds(): Set<string> {
  try {
    if (typeof sessionStorage === 'undefined') return new Set();
    const raw = sessionStorage.getItem(STORAGE_KEY_DISMISSED_NOTIFICATIONS);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

export function recordDismissedNotificationId(runId: string): void {
  try {
    if (typeof sessionStorage === 'undefined') return;
    const set = loadDismissedNotificationIds();
    set.add(runId);
    sessionStorage.setItem(STORAGE_KEY_DISMISSED_NOTIFICATIONS, JSON.stringify(Array.from(set)));
  } catch (err) {
    console.warn('Failed to record dismissed notification ID in sessionStorage:', err);
  }
}

function saveMonitoredRunHint(run: MonitoredRun | null): void {
  try {
    if (typeof sessionStorage === 'undefined') return;
    if (!run) {
      sessionStorage.removeItem(STORAGE_KEY_MONITORED_RUN);
    } else {
      sessionStorage.setItem(
        STORAGE_KEY_MONITORED_RUN,
        JSON.stringify({
          runId: run.runId,
          projectId: run.projectId,
          projectName: run.projectName,
        })
      );
    }
  } catch (err) {
    console.warn('Failed to save monitored run hint to sessionStorage:', err);
  }
}

export function loadMonitoredRunHint(): { runId: string; projectId: string; projectName?: string } | null {
  try {
    if (typeof sessionStorage === 'undefined') return null;
    const raw = sessionStorage.getItem(STORAGE_KEY_MONITORED_RUN);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.runId === 'string' && typeof parsed.projectId === 'string') {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

function dispatchMonitorChange(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent(MONITOR_CHANGE_EVENT, {
        detail: {
          monitoredRun: currentMonitoredRun,
          notification: currentNotification,
        },
      })
    );
  }
}

function dispatchModalChange(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent(MODAL_CHANGE_EVENT, {
        detail: modalState,
      })
    );
  }
}

export function extractRunCounts(run: ResearchRun): {
  gapsDiscovered: number;
  statementsExtracted: number;
  failedPapers: number;
  totalPapers: number;
} {
  let gapsDiscovered = 0;
  let statementsExtracted = 0;
  let failedPapers = 0;
  const totalPapers = Array.isArray(run.papers) ? run.papers.length : (run.summary?.total_papers ?? 0);

  // Summary counts
  if (run.summary) {
    if (typeof run.summary.total_statements === 'number') {
      statementsExtracted = run.summary.total_statements;
    }
    if (typeof run.summary.failed_papers === 'number') {
      failedPapers = run.summary.failed_papers;
    }
    if (typeof run.summary.gaps_discovered === 'number') {
      gapsDiscovered = run.summary.gaps_discovered;
    }
  }

  // Paper-level inspection
  if (Array.isArray(run.papers)) {
    if (statementsExtracted === 0) {
      statementsExtracted = run.papers.reduce((sum, p) => sum + (p.statements_count || 0), 0);
    }
    if (failedPapers === 0) {
      failedPapers = run.papers.filter((p) => p.status === 'FAILED' || p.status === 'ERROR').length;
    }
  }

  // Stage-level details inspection for gaps
  if (gapsDiscovered === 0 && Array.isArray(run.stages)) {
    const gapStage = run.stages.find((s) => s.stage_id === 'GAP_DISCOVERY' || s.name === 'Gap Discovery');
    if (gapStage && (gapStage as any).details && typeof (gapStage as any).details.gaps_discovered === 'number') {
      gapsDiscovered = (gapStage as any).details.gaps_discovered;
    }
  }

  return { gapsDiscovered, statementsExtracted, failedPapers, totalPapers };
}

export function createRunNotification(run: MonitoredRun): RunCompletionNotification {
  const counts = {
    gapsDiscovered: run.gapsDiscovered ?? 0,
    statementsExtracted: run.statementsExtracted ?? 0,
    failedPapers: run.failedPapers ?? 0,
  };

  switch (run.status) {
    case 'AWAITING_REVIEW':
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: 'Research Gaps Ready for Review',
        message:
          counts.gapsDiscovered > 0
            ? `Discovered ${counts.gapsDiscovered} research gap${counts.gapsDiscovered === 1 ? '' : 's'} ready for review and opportunity formulation.`
            : 'Pipeline paused at Awaiting Review: inspect statements and gap evidence.',
        variant: 'warning',
        ...counts,
      };

    case 'COMPLETED':
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: 'Research Pipeline Completed',
        message: `Pipeline finished successfully with ${counts.statementsExtracted} statements extracted across ${run.totalPapers ?? 0} papers.`,
        variant: 'success',
        ...counts,
      };

    case 'PARTIAL_SUCCESS':
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: 'Run Finished with Diagnostic Issues',
        message: `Pipeline finished with ${counts.failedPapers} failed or skipped paper(s). Work has been safely persisted.`,
        variant: 'warning',
        ...counts,
      };

    case 'FAILED':
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: 'Research Pipeline Failed',
        message: run.errorMessage || 'Execution encountered an unexpected failure. Inspect technical logs.',
        variant: 'danger',
        errorMessage: run.errorMessage,
        ...counts,
      };

    case 'INTERRUPTED':
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: 'Research Run Interrupted',
        message: 'Execution was interrupted by a server stop or restart. Use Retry Failed Stages to resume cleanly.',
        variant: 'danger',
        errorMessage: run.errorMessage,
        ...counts,
      };

    default:
      return {
        runId: run.runId,
        projectId: run.projectId,
        projectName: run.projectName,
        status: run.status,
        title: `Run Status: ${run.status}`,
        message: 'Pipeline status updated.',
        variant: 'info',
        ...counts,
      };
  }
}

/**
 * Updates the in-memory monitored run from an authoritative ResearchRun object.
 */
export function reportRunStatus(runData: ResearchRun, projectNameOverride?: string): MonitoredRun {
  const counts = extractRunCounts(runData);

  // Find active stage name
  let currentStage = runData.current_stage || null;
  if (!currentStage && Array.isArray(runData.stages)) {
    const runningStage = runData.stages.find((s) => s.status === 'RUNNING');
    if (runningStage) {
      currentStage = runningStage.name || runningStage.stage_id;
    }
  }

  const updated: MonitoredRun = {
    runId: runData.id,
    projectId: runData.project_id,
    projectName: projectNameOverride || currentMonitoredRun?.projectName,
    status: runData.status,
    currentStage,
    stageProgress: runData.stage_progress,
    gapsDiscovered: counts.gapsDiscovered,
    statementsExtracted: counts.statementsExtracted,
    failedPapers: counts.failedPapers,
    totalPapers: counts.totalPapers,
    errorMessage: runData.error_message,
    lastUpdated: new Date().toISOString(),
  };

  const wasTerminal = currentMonitoredRun ? isTerminalStatus(currentMonitoredRun.status) : false;
  const isNowTerminal = isTerminalStatus(updated.status);

  currentMonitoredRun = updated;
  saveMonitoredRunHint(updated);

  // If newly transitioned to terminal state, trigger notification if not previously dismissed
  if (isNowTerminal) {
    stopBackgroundPolling();
    const dismissedIds = loadDismissedNotificationIds();
    if (!dismissedIds.has(updated.runId) && (!wasTerminal || !currentNotification)) {
      currentNotification = createRunNotification(updated);
    }
  } else if (!isPollingActive && !modalState.isOpen) {
    // If active and modal not open, ensure background polling runs
    startBackgroundPolling();
  }

  dispatchMonitorChange();
  return updated;
}

/**
 * Registers a new or existing run to be actively monitored.
 */
export function registerMonitoredRun(runId: string, projectId: string, projectName?: string): MonitoredRun {
  consecutiveErrors = 0;
  const run: MonitoredRun = {
    runId,
    projectId,
    projectName,
    status: 'RUNNING',
    lastUpdated: new Date().toISOString(),
  };

  currentMonitoredRun = run;
  saveMonitoredRunHint(run);
  dispatchMonitorChange();

  // Fetch immediate authoritative status
  void pollSingleTick();

  return run;
}

/**
 * Dismisses the active completion notification for this run and persists dismissal to sessionStorage.
 */
export function dismissNotification(): void {
  if (currentNotification) {
    recordDismissedNotificationId(currentNotification.runId);
    currentNotification = null;
    dispatchMonitorChange();
  }
}

/**
 * Opens the progress modal for a run.
 */
export function openRunProgress(runId: string, projectId?: string): void {
  modalState = {
    isOpen: true,
    runId,
    projectId: projectId || currentMonitoredRun?.projectId || null,
  };
  // Stop redundant background poll while modal is open and polling
  stopBackgroundPolling();
  dispatchModalChange();
}

/**
 * Closes the progress modal without interrupting background execution.
 */
export function closeRunProgress(): void {
  modalState = {
    isOpen: false,
    runId: modalState.runId,
    projectId: modalState.projectId,
  };
  dispatchModalChange();

  // If the run is still active, resume background polling
  if (currentMonitoredRun && !isTerminalStatus(currentMonitoredRun.status)) {
    startBackgroundPolling();
  }
}

/**
 * Single background poll tick.
 */
async function pollSingleTick(): Promise<void> {
  if (!currentMonitoredRun) {
    stopBackgroundPolling();
    return;
  }

  // If the progress modal is open and polling for this same run, avoid duplicate polling
  if (modalState.isOpen && modalState.runId === currentMonitoredRun.runId) {
    return;
  }

  try {
    const data = await api.getRunStatus(currentMonitoredRun.runId);
    consecutiveErrors = 0;
    reportRunStatus(data);
  } catch (err: any) {
    consecutiveErrors += 1;
    console.warn(`[runMonitor] Poll error (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, err?.message || err);

    // If 404 or missing on backend, clean up stale hint
    if (err?.status === 404 || (err?.message && err.message.includes('not found'))) {
      stopBackgroundPolling();
      currentMonitoredRun = null;
      saveMonitoredRunHint(null);
      dispatchMonitorChange();
      return;
    }

    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
      console.warn('[runMonitor] Max consecutive polling errors reached. Stopping background polling.');
      stopBackgroundPolling();
      return;
    }
  }
}

function scheduleNextPoll(): void {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (!isPollingActive) return;

  pollTimer = setTimeout(async () => {
    if (!isPollingActive) return;
    await pollSingleTick();
    if (isPollingActive && currentMonitoredRun && !isTerminalStatus(currentMonitoredRun.status)) {
      scheduleNextPoll();
    }
  }, BACKGROUND_POLL_INTERVAL_MS);
}

export function startBackgroundPolling(): void {
  if (isPollingActive) return;
  if (!currentMonitoredRun || isTerminalStatus(currentMonitoredRun.status)) return;

  isPollingActive = true;
  scheduleNextPoll();
}

export function stopBackgroundPolling(): void {
  isPollingActive = false;
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

/**
 * Initializes monitoring from sessionStorage recovery hint.
 * Re-fetches status authoritatively from the backend.
 */
export async function initRunMonitor(): Promise<void> {
  if (isInitialized) return;
  isInitialized = true;

  const hint = loadMonitoredRunHint();
  if (!hint?.runId) return;

  // Optimistic initial memory state while re-fetching
  currentMonitoredRun = {
    runId: hint.runId,
    projectId: hint.projectId,
    projectName: hint.projectName,
    status: 'RUNNING',
  };
  dispatchMonitorChange();

  try {
    const authoritativeRun = await api.getRunStatus(hint.runId);
    consecutiveErrors = 0;
    reportRunStatus(authoritativeRun, hint.projectName);
  } catch (err: any) {
    console.warn('[runMonitor] Failed restoring run from sessionStorage hint:', err?.message || err);
    if (err?.status === 404 || (err?.message && err.message.includes('not found'))) {
      currentMonitoredRun = null;
      saveMonitoredRunHint(null);
      dispatchMonitorChange();
    }
  }
}

/**
 * React hook to subscribe to the persistent background run monitor.
 */
export function useRunMonitor(): {
  monitoredRun: MonitoredRun | null;
  notification: RunCompletionNotification | null;
  modalState: ProgressModalState;
  registerMonitoredRun: (runId: string, projectId: string, projectName?: string) => MonitoredRun;
  reportRunStatus: (runData: ResearchRun, projectNameOverride?: string) => MonitoredRun;
  openRunProgress: (runId: string, projectId?: string) => void;
  closeRunProgress: () => void;
  dismissNotification: () => void;
} {
  const [monitoredRun, setMonitoredRun] = useState<MonitoredRun | null>(() => currentMonitoredRun);
  const [notification, setNotification] = useState<RunCompletionNotification | null>(() => currentNotification);
  const [currentModalState, setCurrentModalState] = useState<ProgressModalState>(() => modalState);

  useEffect(() => {
    // Initial restoration
    void initRunMonitor();

    const handleMonitorChange = (e: Event) => {
      const custom = e as CustomEvent<{ monitoredRun: MonitoredRun | null; notification: RunCompletionNotification | null }>;
      if (custom.detail) {
        setMonitoredRun(custom.detail.monitoredRun);
        setNotification(custom.detail.notification);
      } else {
        setMonitoredRun(currentMonitoredRun);
        setNotification(currentNotification);
      }
    };

    const handleModalChange = (e: Event) => {
      const custom = e as CustomEvent<ProgressModalState>;
      if (custom.detail) {
        setCurrentModalState(custom.detail);
      } else {
        setCurrentModalState(modalState);
      }
    };

    window.addEventListener(MONITOR_CHANGE_EVENT, handleMonitorChange);
    window.addEventListener(MODAL_CHANGE_EVENT, handleModalChange);

    return () => {
      window.removeEventListener(MONITOR_CHANGE_EVENT, handleMonitorChange);
      window.removeEventListener(MODAL_CHANGE_EVENT, handleModalChange);
    };
  }, []);

  return {
    monitoredRun,
    notification,
    modalState: currentModalState,
    registerMonitoredRun,
    reportRunStatus,
    openRunProgress,
    closeRunProgress,
    dismissNotification,
  };
}

/**
 * Test helper to reset internal module state between unit tests.
 */
export function _resetMonitorStateForTesting(): void {
  stopBackgroundPolling();
  currentMonitoredRun = null;
  currentNotification = null;
  modalState = { isOpen: false, runId: null, projectId: null };
  consecutiveErrors = 0;
  isPollingActive = false;
  isInitialized = false;
  if (typeof sessionStorage !== 'undefined') {
    sessionStorage.removeItem(STORAGE_KEY_MONITORED_RUN);
    sessionStorage.removeItem(STORAGE_KEY_DISMISSED_NOTIFICATIONS);
  }
}
