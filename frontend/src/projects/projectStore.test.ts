import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Mock Web Storage & DOM APIs in Node environment before importing store
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

const dispatchedEvents: { type: string; detail: unknown }[] = [];
Object.defineProperty(globalThis, 'window', {
  value: {
    dispatchEvent: (evt: { type: string; detail: unknown }) => {
      dispatchedEvents.push(evt);
      return true;
    },
    addEventListener: () => {},
    removeEventListener: () => {},
  },
  writable: true,
});

// Import modules under test
import { api } from '../api/client.ts';
import {
  DEFAULT_BENCHMARK_PROJECT,
  getProjects,
  getActiveProjectId,
  setActiveProjectId,
  deleteProject,
  createProject,
  syncProjectsWithServer,
  cleanupProjectBrowserStorage,
  loadDeletedProjectIds,
  recordDeletedProjectId,
  clearDeletedProjectId,
} from './projectStore.ts';
import type { ProjectDeleteResponse, ProjectRecord } from '../types.ts';

describe('Project Deletion & Safe Store Cleanup', () => {
  const STORAGE_KEY_PROJECTS = 'ideagen_projects_v1';
  const STORAGE_KEY_ACTIVE_PROJECT = 'ideagen_active_project_id';

  beforeEach(() => {
    mockLocalStorage.clear();
    mockSessionStorage.clear();
    dispatchedEvents.length = 0;

    // Reset default benchmark project in storage
    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT])
    );
    mockLocalStorage.setItem(STORAGE_KEY_ACTIVE_PROJECT, DEFAULT_BENCHMARK_PROJECT.id);
  });

  test('1. Successful deletion removes the project from the server-backed local list', async () => {
    const customProject = {
      id: 'proj_alpha_123',
      name: 'Alpha Project',
      topic: 'Topic Alpha',
      description: 'Alpha description',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, customProject])
    );

    // Mock API client deleteProject
    let apiCalledWith: string | null = null;
    const mockResponse: ProjectDeleteResponse = {
      project_id: customProject.id,
      deleted: true,
      deleted_at: new Date().toISOString(),
      summary: {
        runs_deleted: 2,
        run_papers_deleted: 5,
        selections_deleted: 3,
        paper_links_removed: 4,
        gaps_deleted: 1,
        gap_evidence_deleted: 2,
        research_opportunities_deleted: 1,
        prior_art_assessments_deleted: 0,
        refinement_iterations_deleted: 0,
        lifecycle_opportunities_deleted: 1,
        opportunity_evidence_deleted: 2,
        opportunity_history_deleted: 1,
      },
    };

    const originalDelete = api.deleteProject;
    api.deleteProject = async (id: string) => {
      apiCalledWith = id;
      return mockResponse;
    };

    try {
      const result = await deleteProject(customProject.id);

      assert.equal(apiCalledWith, customProject.id);
      assert.deepEqual(result, mockResponse);

      const remaining = getProjects();
      assert.equal(remaining.length, 1);
      assert.equal(remaining[0].id, DEFAULT_BENCHMARK_PROJECT.id);
      assert.equal(remaining.some((p) => p.id === customProject.id), false);
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('2. Failed deletion preserves the project and its browser state', async () => {
    const customProject = {
      id: 'proj_beta_456',
      name: 'Beta Project',
      topic: 'Topic Beta',
      description: 'Beta description',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, customProject])
    );
    mockSessionStorage.setItem(`ideagen_search_state_${customProject.id}`, JSON.stringify({ query: 'quantum' }));

    const originalDelete = api.deleteProject;
    api.deleteProject = async () => {
      throw new Error('Project not found (HTTP 404)');
    };

    try {
      await assert.rejects(
        async () => {
          await deleteProject(customProject.id);
        },
        {
          name: 'Error',
          message: 'Project not found (HTTP 404)',
        }
      );

      // Verify state is completely preserved
      const remaining = getProjects();
      assert.equal(remaining.length, 2);
      assert.equal(remaining.some((p) => p.id === customProject.id), true);

      // Verify search context is untouched
      assert.notEqual(
        mockSessionStorage.getItem(`ideagen_search_state_${customProject.id}`),
        null
      );
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('3. Deleting the active project selects the safe fallback or dashboard', async () => {
    const customProject = {
      id: 'proj_gamma_789',
      name: 'Gamma Project',
      topic: 'Topic Gamma',
      description: 'Gamma description',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, customProject])
    );
    setActiveProjectId(customProject.id);
    assert.equal(getActiveProjectId(), customProject.id);

    const originalDelete = api.deleteProject;
    api.deleteProject = async (id: string) => ({
      project_id: id,
      deleted: true,
      deleted_at: new Date().toISOString(),
      summary: {} as any,
    });

    try {
      await deleteProject(customProject.id);

      // Active project must fall back to DEFAULT_BENCHMARK_PROJECT
      assert.equal(getActiveProjectId(), DEFAULT_BENCHMARK_PROJECT.id);
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('4. Deleting a non-active project leaves the active project unchanged', async () => {
    const nonActiveProject = {
      id: 'proj_non_active',
      name: 'Non Active Project',
      topic: 'Topic Non Active',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, nonActiveProject])
    );
    // Active project is benchmark
    setActiveProjectId(DEFAULT_BENCHMARK_PROJECT.id);

    const originalDelete = api.deleteProject;
    api.deleteProject = async (id: string) => ({
      project_id: id,
      deleted: true,
      deleted_at: new Date().toISOString(),
      summary: {} as any,
    });

    try {
      await deleteProject(nonActiveProject.id);

      assert.equal(getActiveProjectId(), DEFAULT_BENCHMARK_PROJECT.id);
      const remaining = getProjects();
      assert.equal(remaining.some((p) => p.id === nonActiveProject.id), false);
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('5. Project-specific localStorage and sessionStorage keys are removed without clearing other projects or global state', () => {
    const projId = 'proj_cleanup_test';
    const otherProjId = 'proj_other_stay';

    // Populate keys for target project
    mockSessionStorage.setItem(`ideagen_search_state_${projId}`, JSON.stringify({ query: 'rag' }));
    mockLocalStorage.setItem(`ideagen_search_state_${projId}`, JSON.stringify({ query: 'rag' }));
    mockLocalStorage.setItem(`ideagen_project_${projId}`, 'custom_data');

    // Populate keys for other project
    mockSessionStorage.setItem(`ideagen_search_state_${otherProjId}`, JSON.stringify({ query: 'agent' }));
    mockLocalStorage.setItem(`ideagen_search_state_${otherProjId}`, JSON.stringify({ query: 'agent' }));

    // Global keys
    mockLocalStorage.setItem('ideagen_projects_v1', 'projects_data');
    mockLocalStorage.setItem('ideagen_active_project_id', otherProjId);

    cleanupProjectBrowserStorage(projId);

    // Target keys removed
    assert.equal(mockSessionStorage.getItem(`ideagen_search_state_${projId}`), null);
    assert.equal(mockLocalStorage.getItem(`ideagen_search_state_${projId}`), null);
    assert.equal(mockLocalStorage.getItem(`ideagen_project_${projId}`), null);

    // Other project keys preserved
    assert.notEqual(mockSessionStorage.getItem(`ideagen_search_state_${otherProjId}`), null);
    assert.notEqual(mockLocalStorage.getItem(`ideagen_search_state_${otherProjId}`), null);

    // System keys preserved
    assert.equal(mockLocalStorage.getItem('ideagen_projects_v1'), 'projects_data');
    assert.equal(mockLocalStorage.getItem('ideagen_active_project_id'), otherProjId);
  });

  test('6. Protected rag-benchmark rejects deletion in the store', async () => {
    await assert.rejects(
      async () => {
        await deleteProject(DEFAULT_BENCHMARK_PROJECT.id);
      },
      {
        name: 'Error',
        message: "The primary benchmark project 'rag-benchmark' cannot be deleted.",
      }
    );

    const projs = getProjects();
    assert.equal(projs.some((p) => p.id === DEFAULT_BENCHMARK_PROJECT.id), true);
  });

  test('7. A 409 active-work response leaves state intact and displays a useful error', async () => {
    const busyProject = {
      id: 'proj_busy_run',
      name: 'Busy Project',
      topic: 'Topic Busy',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, busyProject])
    );

    const originalDelete = api.deleteProject;
    api.deleteProject = async (id: string) => {
      throw new Error(`Cannot delete project '${id}': an active research run or opportunity generation is currently in progress. (HTTP 409)`);
    };

    try {
      await assert.rejects(
        async () => {
          await deleteProject(busyProject.id);
        },
        {
          name: 'Error',
          message: `Cannot delete project '${busyProject.id}': an active research run or opportunity generation is currently in progress. (HTTP 409)`,
        }
      );

      const projs = getProjects();
      assert.equal(projs.some((p) => p.id === busyProject.id), true);
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('8. Duplicate submission is prevented while the request is pending', async () => {
    const slowProject = {
      id: 'proj_slow_del',
      name: 'Slow Project',
      topic: 'Topic Slow',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, slowProject])
    );

    let resolvePromise: (val: ProjectDeleteResponse) => void;
    const pendingPromise = new Promise<ProjectDeleteResponse>((resolve) => {
      resolvePromise = resolve;
    });

    const originalDelete = api.deleteProject;
    api.deleteProject = async () => pendingPromise;

    try {
      // First deletion starts
      const firstCall = deleteProject(slowProject.id);

      // Second immediate concurrent deletion should be rejected
      await assert.rejects(
        async () => {
          await deleteProject(slowProject.id);
        },
        {
          name: 'Error',
          message: `Deletion of project '${slowProject.id}' is already in progress.`,
        }
      );

      // Complete first call
      resolvePromise!({
        project_id: slowProject.id,
        deleted: true,
        deleted_at: new Date().toISOString(),
        summary: {} as any,
      });

      await firstCall;
      const projs = getProjects();
      assert.equal(projs.some((p) => p.id === slowProject.id), false);
    } finally {
      api.deleteProject = originalDelete;
    }
  });

  test('9. Subsequent project synchronization does not resurrect a deleted project', async () => {
    const deletedProjId = 'proj_to_delete_sync';
    const projToDelete = {
      id: deletedProjId,
      name: 'To Delete Sync',
      topic: 'Topic Sync',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, projToDelete])
    );

    const originalDelete = api.deleteProject;
    const originalList = api.listProjects;
    const originalCreate = api.createProject;

    api.deleteProject = async (id: string) => ({
      project_id: id,
      deleted: true,
      deleted_at: new Date().toISOString(),
      summary: {} as any,
    });

    let createAttemptedWith: string | null = null;
    api.createProject = async (body: any) => {
      createAttemptedWith = body.id;
      return body as ProjectRecord;
    };

    try {
      // 1. Delete project
      await deleteProject(deletedProjId);

      // 2. Simulate server list returning benchmark only (or even if an in-flight server list had it)
      api.listProjects = async () => [
        {
          id: DEFAULT_BENCHMARK_PROJECT.id,
          title: DEFAULT_BENCHMARK_PROJECT.name,
          topic: DEFAULT_BENCHMARK_PROJECT.topic,
          description: DEFAULT_BENCHMARK_PROJECT.description,
          temporal_cutoff: DEFAULT_BENCHMARK_PROJECT.temporalCutoff,
          status: 'ACTIVE',
        } as ProjectRecord,
      ];

      // 3. Run sync
      await syncProjectsWithServer();

      // Verify createProject was NEVER called for the deleted project
      assert.equal(createAttemptedWith, null);

      // Verify local storage does not contain the deleted project
      const projs = getProjects();
      assert.equal(projs.some((p) => p.id === deletedProjId), false);
    } finally {
      api.deleteProject = originalDelete;
      api.listProjects = originalList;
      api.createProject = originalCreate;
    }
  });

  test('10. Exact project-scoped storage cleanup preserves overlapping prefix project keys (proj_1 vs proj_1_eval)', () => {
    const projA = 'proj_1';
    const projB = 'proj_1_eval';

    mockSessionStorage.setItem(`ideagen_search_state_${projA}`, 'search_A');
    mockLocalStorage.setItem(`ideagen_search_state_${projA}`, 'search_A');
    mockLocalStorage.setItem(`ideagen_project_${projA}`, 'data_A');

    mockSessionStorage.setItem(`ideagen_search_state_${projB}`, 'search_B');
    mockLocalStorage.setItem(`ideagen_search_state_${projB}`, 'search_B');
    mockLocalStorage.setItem(`ideagen_project_${projB}`, 'data_B');

    mockLocalStorage.setItem(STORAGE_KEY_PROJECTS, 'projects');
    mockLocalStorage.setItem(STORAGE_KEY_ACTIVE_PROJECT, 'active');

    cleanupProjectBrowserStorage(projA);

    // proj_1 keys must be removed
    assert.equal(mockSessionStorage.getItem(`ideagen_search_state_${projA}`), null);
    assert.equal(mockLocalStorage.getItem(`ideagen_search_state_${projA}`), null);
    assert.equal(mockLocalStorage.getItem(`ideagen_project_${projA}`), null);

    // proj_1_eval keys must be strictly preserved
    assert.equal(mockSessionStorage.getItem(`ideagen_search_state_${projB}`), 'search_B');
    assert.equal(mockLocalStorage.getItem(`ideagen_search_state_${projB}`), 'search_B');
    assert.equal(mockLocalStorage.getItem(`ideagen_project_${projB}`), 'data_B');

    // Global keys preserved
    assert.equal(mockLocalStorage.getItem(STORAGE_KEY_PROJECTS), 'projects');
    assert.equal(mockLocalStorage.getItem(STORAGE_KEY_ACTIVE_PROJECT), 'active');
  });

  test('11. Tombstones are visible across shared localStorage and prevent resurrection across simulated tabs', async () => {
    const deletedId = 'proj_tab_cross_sync';
    const project = {
      id: deletedId,
      name: 'Cross Tab Project',
      topic: 'Topic',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    // Tab 1 records deletion tombstone in localStorage
    recordDeletedProjectId(deletedId);
    assert.equal(loadDeletedProjectIds().has(deletedId), true);

    // Tab 2 has a stale local project list in its in-memory snapshot
    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, project])
    );

    let createAttemptedForDeleted = false;
    const originalCreate = api.createProject;
    const originalList = api.listProjects;
    api.createProject = async (body: any) => {
      if (body.id === deletedId) {
        createAttemptedForDeleted = true;
      }
      return body;
    };
    api.listProjects = async () => [
      {
        id: DEFAULT_BENCHMARK_PROJECT.id,
        title: DEFAULT_BENCHMARK_PROJECT.name,
        topic: DEFAULT_BENCHMARK_PROJECT.topic,
        status: 'ACTIVE',
      } as ProjectRecord,
    ];

    try {
      // Tab 2 runs sync
      await syncProjectsWithServer();

      // createProject must not have been called because tombstone in shared localStorage stopped it
      assert.equal(createAttemptedForDeleted, false);
      assert.equal(getProjects().some((p) => p.id === deletedId), false);
    } finally {
      api.createProject = originalCreate;
      api.listProjects = originalList;
    }
  });

  test('12. Stale sync response containing deleted project does not resurrect it', async () => {
    const deletedId = 'proj_stale_response';
    recordDeletedProjectId(deletedId);

    const originalList = api.listProjects;
    // Simulate server returning stale record for the deleted project (e.g. read replica delay)
    api.listProjects = async () => [
      {
        id: DEFAULT_BENCHMARK_PROJECT.id,
        title: DEFAULT_BENCHMARK_PROJECT.name,
        topic: DEFAULT_BENCHMARK_PROJECT.topic,
        temporal_cutoff: DEFAULT_BENCHMARK_PROJECT.temporalCutoff,
        status: 'ACTIVE',
      } as ProjectRecord,
      {
        id: deletedId,
        title: 'Stale Deleted Project',
        topic: 'Stale Topic',
        status: 'ACTIVE',
      } as ProjectRecord,
    ];

    try {
      await syncProjectsWithServer();
      const projects = getProjects();
      assert.equal(projects.some((p) => p.id === deletedId), false);
    } finally {
      api.listProjects = originalList;
    }
  });

  test('13. Legitimate explicit creation with the same ID clears the tombstone', async () => {
    const targetId = 'proj_recreate_test';
    recordDeletedProjectId(targetId);
    assert.equal(loadDeletedProjectIds().has(targetId), true);

    // Explicit recreation clears tombstone
    clearDeletedProjectId(targetId);
    assert.equal(loadDeletedProjectIds().has(targetId), false);

    // Also verify createProject removes tombstone when server returns that record
    recordDeletedProjectId('proj_created_clears');
    const originalCreate = api.createProject;
    api.createProject = async (body: any) => ({
      ...body,
      id: 'proj_created_clears',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      status: 'ACTIVE',
    });

    try {
      await createProject({ name: 'Recreated', topic: 'Topic' });
      assert.equal(loadDeletedProjectIds().has('proj_created_clears'), false);
    } finally {
      api.createProject = originalCreate;
    }
  });

  test('14. Storage failure during tombstone recording preserves backend deletion while surfacing error', async () => {
    const customProject = {
      id: 'proj_storage_err_test',
      name: 'Storage Error Project',
      topic: 'Topic',
      description: '',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ACTIVE' as const,
      isDefaultBenchmark: false,
    };

    mockLocalStorage.setItem(
      STORAGE_KEY_PROJECTS,
      JSON.stringify([DEFAULT_BENCHMARK_PROJECT, customProject])
    );

    const originalDelete = api.deleteProject;
    api.deleteProject = async (id: string) => ({
      project_id: id,
      deleted: true,
      deleted_at: new Date().toISOString(),
      summary: {} as any,
    });

    // Mock localStorage.setItem to throw when writing tombstone
    const originalSetItem = mockLocalStorage.setItem.bind(mockLocalStorage);
    mockLocalStorage.setItem = (key: string, value: string) => {
      if (key === 'ideagen_deleted_projects_v1') {
        throw new Error('QuotaExceededError: storage full');
      }
      return originalSetItem(key, value);
    };

    try {
      await assert.rejects(
        async () => {
          await deleteProject(customProject.id);
        },
        {
          name: 'Error',
          message: 'Project was deleted on the server, but cross-tab synchronization storage failed: QuotaExceededError: storage full',
        }
      );

      // Local state was still cleaned up and updated
      assert.equal(getProjects().some((p) => p.id === customProject.id), false);
    } finally {
      api.deleteProject = originalDelete;
      mockLocalStorage.setItem = originalSetItem;
    }
  });
});

describe('api.deleteProject API Client', () => {
  const originalFetch = globalThis.fetch;

  test('calls DELETE /api/v1/projects/{projectId} and parses ProjectDeleteResponse', async () => {
    let capturedUrl = '';
    let capturedMethod = '';

    const mockResponsePayload: ProjectDeleteResponse = {
      project_id: 'proj_spec_test',
      deleted: true,
      deleted_at: '2026-10-25T01:00:00Z',
      summary: {
        runs_deleted: 1,
        run_papers_deleted: 2,
        selections_deleted: 0,
        paper_links_removed: 1,
        gaps_deleted: 0,
        gap_evidence_deleted: 0,
        research_opportunities_deleted: 0,
        prior_art_assessments_deleted: 0,
        refinement_iterations_deleted: 0,
        lifecycle_opportunities_deleted: 0,
        opportunity_evidence_deleted: 0,
        opportunity_history_deleted: 0,
      },
    };

    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      capturedUrl = String(input);
      capturedMethod = init?.method || 'GET';
      return new Response(JSON.stringify(mockResponsePayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }) as any;

    try {
      const res = await api.deleteProject('proj_spec_test');
      assert.equal(capturedUrl, '/api/v1/projects/proj_spec_test');
      assert.equal(capturedMethod, 'DELETE');
      assert.deepEqual(res, mockResponsePayload);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('encodes URI component in project_id', async () => {
    let capturedUrl = '';

    globalThis.fetch = (async (input: RequestInfo | URL) => {
      capturedUrl = String(input);
      return new Response(JSON.stringify({ project_id: 'a/b c', deleted: true, deleted_at: '', summary: {} }), {
        status: 200,
      });
    }) as any;

    try {
      await api.deleteProject('a/b c');
      assert.equal(capturedUrl, '/api/v1/projects/a%2Fb%20c');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles 400 protected project error with detail', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        JSON.stringify({ detail: "The primary benchmark project 'rag-benchmark' cannot be deleted." }),
        { status: 400 }
      );
    }) as any;

    try {
      await assert.rejects(
        async () => {
          await api.deleteProject('rag-benchmark');
        },
        {
          name: 'Error',
          message: "The primary benchmark project 'rag-benchmark' cannot be deleted. (HTTP 400)",
        }
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles 404 missing project error with detail', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        JSON.stringify({ detail: "Project 'proj_non_existent' not found." }),
        { status: 404 }
      );
    }) as any;

    try {
      await assert.rejects(
        async () => {
          await api.deleteProject('proj_non_existent');
        },
        {
          name: 'Error',
          message: "Project 'proj_non_existent' not found. (HTTP 404)",
        }
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('handles 409 active work error with detail', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        JSON.stringify({
          detail: "Cannot delete project 'proj_active': an active research run or opportunity generation is currently in progress.",
        }),
        { status: 409 }
      );
    }) as any;

    try {
      await assert.rejects(
        async () => {
          await api.deleteProject('proj_active');
        },
        {
          name: 'Error',
          message: "Cannot delete project 'proj_active': an active research run or opportunity generation is currently in progress. (HTTP 409)",
        }
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
