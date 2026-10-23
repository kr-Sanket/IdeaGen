import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { ProjectRecord, ResearchProject } from '../types';

const STORAGE_KEY_PROJECTS = 'ideagen_projects_v1';
const STORAGE_KEY_ACTIVE_PROJECT = 'ideagen_active_project_id';
const PROJECT_CHANGE_EVENT = 'ideagen_project_change';

export const DEFAULT_BENCHMARK_PROJECT: ResearchProject = {
  id: 'rag-benchmark',
  name: 'Retrieval-Augmented Generation (RAG)',
  topic: 'Evidence-grounded discovery, gap analysis, and temporal evolution of RAG architectures',
  description:
    'Core benchmark project analyzing pre-2021 foundation papers against 2023–2026 evaluation literature with Ollama Qwen 3 (8B) and Nomic Embed Text.',
  createdAt: '2026-10-19T13:41:49.000Z',
  updatedAt: '2026-10-22T07:03:48.000Z',
  status: 'ACTIVE',
  temporalCutoff: '2020-12-31',
  isDefaultBenchmark: true,
};

function loadStoredProjects(): ResearchProject[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_PROJECTS);
    if (!raw) return [DEFAULT_BENCHMARK_PROJECT];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return [DEFAULT_BENCHMARK_PROJECT];
    }
    // Guarantee default benchmark project is present
    const hasBenchmark = parsed.some((p) => p.id === DEFAULT_BENCHMARK_PROJECT.id);
    if (!hasBenchmark) {
      return [DEFAULT_BENCHMARK_PROJECT, ...parsed];
    }
    return parsed;
  } catch {
    return [DEFAULT_BENCHMARK_PROJECT];
  }
}

function saveProjects(projects: ResearchProject[]): void {
  try {
    localStorage.setItem(STORAGE_KEY_PROJECTS, JSON.stringify(projects));
  } catch (err) {
    console.warn('Failed to save projects to localStorage:', err);
  }
}

export function getProjects(): ResearchProject[] {
  return loadStoredProjects();
}

export function getProject(id: string): ResearchProject | undefined {
  const projects = loadStoredProjects();
  return projects.find((p) => p.id === id);
}

export function getActiveProjectId(): string {
  try {
    const active = localStorage.getItem(STORAGE_KEY_ACTIVE_PROJECT);
    if (active && getProject(active)) {
      return active;
    }
  } catch {
    // Ignore localStorage errors
  }
  return DEFAULT_BENCHMARK_PROJECT.id;
}

export function setActiveProjectId(id: string): void {
  try {
    localStorage.setItem(STORAGE_KEY_ACTIVE_PROJECT, id);
    window.dispatchEvent(new CustomEvent(PROJECT_CHANGE_EVENT, { detail: { activeProjectId: id } }));
  } catch (err) {
    console.warn('Failed to set active project id:', err);
  }
}

function fromRecord(record: ProjectRecord, local?: ResearchProject): ResearchProject {
  return {
    id: record.id,
    name: record.title,
    // An empty server field never erases a value this browser still has (an older backend did not
    // return topic/cutoff; the next sync sends the local value, which the server fills in).
    topic: record.topic || local?.topic || record.title,
    description: record.description || local?.description || '',
    createdAt: record.created_at ?? local?.createdAt ?? new Date().toISOString(),
    updatedAt: record.updated_at ?? local?.updatedAt ?? new Date().toISOString(),
    status: record.status === 'ARCHIVED' ? 'ARCHIVED' : 'ACTIVE',
    temporalCutoff: record.temporal_cutoff || local?.temporalCutoff || undefined,
    isDefaultBenchmark: record.id === DEFAULT_BENCHMARK_PROJECT.id,
  };
}

function toCreateRequest(project: ResearchProject) {
  return {
    id: project.id,
    title: project.name,
    topic: project.topic || undefined,
    description: project.description || undefined,
    temporal_cutoff: project.temporalCutoff || undefined,
  };
}

let syncInFlight: Promise<void> | null = null;

/**
 * Makes the backend the source of truth for projects. Projects that exist only in this browser
 * (created before projects were stored server-side) are registered under their existing IDs, so
 * papers added to them can be associated; projects known to the server are added locally.
 * Rejects if the backend cannot be reached; the local cache is then left unchanged.
 */
export function syncProjectsWithServer(): Promise<void> {
  if (syncInFlight) return syncInFlight;
  syncInFlight = (async () => {
    const serverProjects = await api.listProjects();
    const serverById = new Map(serverProjects.map((p) => [p.id, p]));
    const local = loadStoredProjects();

    for (const project of local) {
      const record = serverById.get(project.id);
      // Missing on the server, or registered by an older backend that dropped topic/cutoff:
      // the create call is idempotent and only fills fields the server record lacks.
      const lacksDetails =
        record !== undefined &&
        ((record.temporal_cutoff == null && Boolean(project.temporalCutoff)) ||
          ((!record.topic || record.topic === record.title) && Boolean(project.topic) && project.topic !== project.name));
      if (record === undefined || lacksDetails) {
        try {
          serverById.set(project.id, await api.createProject(toCreateRequest(project)));
        } catch (err) {
          // One rejected project (e.g. invalid legacy data) must not block syncing the others;
          // it stays local-only and is retried on the next sync.
          console.warn(`Could not register project ${project.id} with the backend:`, err);
        }
      }
    }

    const localIds = new Set(local.map((p) => p.id));
    const merged = [
      ...local.map((p) => {
        const record = serverById.get(p.id);
        return record ? fromRecord(record, p) : p;
      }),
      ...serverProjects.filter((p) => !localIds.has(p.id)).map((p) => fromRecord(p)),
    ];
    saveProjects(merged);
    window.dispatchEvent(new CustomEvent(PROJECT_CHANGE_EVENT, { detail: { projects: merged } }));
  })().finally(() => {
    syncInFlight = null;
  });
  return syncInFlight;
}

/** Creates the project on the backend first; nothing is stored locally if that fails. */
export async function createProject(params: {
  name: string;
  topic: string;
  description?: string;
  temporalCutoff?: string;
}): Promise<ResearchProject> {
  const id = `proj_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const record = await api.createProject({
    id,
    title: params.name.trim(),
    topic: params.topic.trim(),
    description: (params.description || '').trim() || undefined,
    temporal_cutoff: params.temporalCutoff || undefined,
  });

  const newProject = fromRecord(record);
  const updated = [newProject, ...loadStoredProjects().filter((p) => p.id !== newProject.id)];
  saveProjects(updated);
  setActiveProjectId(newProject.id);
  window.dispatchEvent(new CustomEvent(PROJECT_CHANGE_EVENT, { detail: { projects: updated, activeProjectId: newProject.id } }));
  return newProject;
}

export function updateProject(id: string, updates: Partial<ResearchProject>): ResearchProject | null {
  const projects = loadStoredProjects();
  const index = projects.findIndex((p) => p.id === id);
  if (index === -1) return null;

  const updatedProj = {
    ...projects[index],
    ...updates,
    updatedAt: new Date().toISOString(),
  };
  projects[index] = updatedProj;
  saveProjects(projects);
  window.dispatchEvent(new CustomEvent(PROJECT_CHANGE_EVENT, { detail: { projects } }));
  return updatedProj;
}

/** Removes a project from this browser only. Server-side project deletion is not implemented. */
export function deleteProject(id: string): boolean {
  if (id === DEFAULT_BENCHMARK_PROJECT.id) {
    // Cannot delete primary benchmark project
    return false;
  }
  const projects = loadStoredProjects();
  const filtered = projects.filter((p) => p.id !== id);
  saveProjects(filtered);

  if (getActiveProjectId() === id) {
    setActiveProjectId(DEFAULT_BENCHMARK_PROJECT.id);
  } else {
    window.dispatchEvent(new CustomEvent(PROJECT_CHANGE_EVENT, { detail: { projects: filtered } }));
  }
  return true;
}

export function useProjectContext(): {
  projects: ResearchProject[];
  activeProject: ResearchProject;
  activeProjectId: string;
  setActiveProject: (id: string) => void;
  createProject: (params: { name: string; topic: string; description?: string; temporalCutoff?: string }) => Promise<ResearchProject>;
  deleteProject: (id: string) => boolean;
} {
  const [projects, setProjects] = useState<ResearchProject[]>(() => loadStoredProjects());
  const [activeId, setActiveId] = useState<string>(() => getActiveProjectId());

  const sync = useCallback(() => {
    const projs = loadStoredProjects();
    setProjects(projs);
    const currActive = getActiveProjectId();
    setActiveId(currActive);
  }, []);

  useEffect(() => {
    const handleCustomChange = () => sync();
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY_PROJECTS || e.key === STORAGE_KEY_ACTIVE_PROJECT) {
        sync();
      }
    };

    window.addEventListener(PROJECT_CHANGE_EVENT, handleCustomChange);
    window.addEventListener('storage', handleStorage);
    // Initial sync; if the backend is down, the Navbar health poll re-syncs once it is reachable.
    syncProjectsWithServer().catch((err) => console.warn('Project sync with backend failed:', err));
    return () => {
      window.removeEventListener(PROJECT_CHANGE_EVENT, handleCustomChange);
      window.removeEventListener('storage', handleStorage);
    };
  }, [sync]);

  const activeProject = projects.find((p) => p.id === activeId) || DEFAULT_BENCHMARK_PROJECT;

  const handleSetActive = useCallback((id: string) => {
    setActiveProjectId(id);
    setActiveId(id);
  }, []);

  const handleCreate = useCallback(async (params: { name: string; topic: string; description?: string; temporalCutoff?: string }) => {
    const p = await createProject(params);
    setProjects(loadStoredProjects());
    setActiveId(p.id);
    return p;
  }, []);

  const handleDelete = useCallback((id: string) => {
    const ok = deleteProject(id);
    if (ok) {
      setProjects(loadStoredProjects());
      setActiveId(getActiveProjectId());
    }
    return ok;
  }, []);

  return {
    projects,
    activeProject,
    activeProjectId: activeId,
    setActiveProject: handleSetActive,
    createProject: handleCreate,
    deleteProject: handleDelete,
  };
}
