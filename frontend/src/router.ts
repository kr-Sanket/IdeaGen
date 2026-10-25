import { useState, useEffect, useCallback } from 'react';
import { getActiveProjectId } from './projects/projectStore';

export type RouteName =
  | 'dashboard'
  | 'workspaces'
  | 'settings'
  | 'project-overview'
  | 'project-papers'
  | 'project-gaps'
  | 'project-opportunities'
  | 'project-opportunity-detail'
  | 'project-evolution'
  | 'project-history'
  // Backward compatibility aliases
  | 'search'
  | 'gaps'
  | 'overview'
  | 'opportunities'
  | 'opportunity-detail'
  | 'corpus'
  | 'evolution'
  | 'history';

export interface ParsedRoute {
  name: RouteName;
  params: Record<string, string>;
  rawHash: string;
}

export function parseHash(hash: string): ParsedRoute {
  const clean = hash.replace(/^#\/?/, '');
  const [pathPart, queryPart] = clean.split('?');
  const segments = pathPart ? pathPart.split('/').filter(Boolean) : [];
  const searchParams = new URLSearchParams(queryPart || '');
  const params: Record<string, string> = {};

  searchParams.forEach((val, key) => {
    params[key] = val;
  });

  // Default landing page is Dashboard
  if (segments.length === 0 || segments[0] === 'dashboard') {
    return { name: 'dashboard', params, rawHash: hash };
  }

  // Global top-level views
  if (segments[0] === 'workspaces') {
    return { name: 'workspaces', params, rawHash: hash };
  }

  if (segments[0] === 'settings') {
    return { name: 'settings', params, rawHash: hash };
  }

  // Project-scoped routes: #/projects/:projectId/...
  if (segments[0] === 'projects' && segments.length >= 2) {
    const projectId = segments[1];
    params['projectId'] = projectId;
    const sub = segments[2] || 'overview';

    if (sub === 'overview') {
      return { name: 'project-overview', params, rawHash: hash };
    }
    if (sub === 'papers') {
      if (segments.length > 3 && segments[3]) {
        params['paper_id'] = segments[3];
      }
      return { name: 'project-papers', params, rawHash: hash };
    }
    if (sub === 'gaps') {
      if (segments.length > 3 && segments[3]) {
        params['gap_id'] = segments[3];
      }
      return { name: 'project-gaps', params, rawHash: hash };
    }
    if (sub === 'opportunities') {
      if (segments.length > 3 && segments[3]) {
        return {
          name: 'project-opportunity-detail',
          params: { ...params, id: segments[3] },
          rawHash: hash,
        };
      }
      return { name: 'project-opportunities', params, rawHash: hash };
    }
    if (sub === 'evolution') {
      return { name: 'project-evolution', params, rawHash: hash };
    }
    if (sub === 'history') {
      return { name: 'project-history', params, rawHash: hash };
    }
    return { name: 'project-overview', params, rawHash: hash };
  }

  // Backward-compatible shortcut routes mapping to active project context
  const activeProjectId = getActiveProjectId();
  if (!params['projectId']) {
    params['projectId'] = activeProjectId;
  }

  if (segments[0] === 'overview' || segments[0] === 'history') {
    return { name: segments[0] === 'history' ? 'project-history' : 'project-overview', params, rawHash: hash };
  }

  if (segments[0] === 'papers' || segments[0] === 'search' || segments[0] === 'corpus') {
    if (segments[0] === 'corpus') {
      params['tab'] = 'corpus';
    } else if (segments[0] === 'search') {
      params['tab'] = 'search';
    }
    if (segments.length > 1 && segments[1]) {
      params['paper_id'] = segments[1];
    }
    return { name: 'project-papers', params, rawHash: hash };
  }

  if (segments[0] === 'gaps') {
    if (segments.length > 1 && segments[1]) {
      params['gap_id'] = segments[1];
    }
    return { name: 'project-gaps', params, rawHash: hash };
  }

  if (segments[0] === 'opportunities') {
    if (segments.length > 1 && segments[1]) {
      return {
        name: 'project-opportunity-detail',
        params: { ...params, id: segments[1] },
        rawHash: hash,
      };
    }
    return { name: 'project-opportunities', params, rawHash: hash };
  }

  if (segments[0] === 'evolution') {
    return { name: 'project-evolution', params, rawHash: hash };
  }

  return { name: 'dashboard', params, rawHash: hash };
}

export function navigate(path: string): void {
  const targetHash = path.startsWith('#') ? path : `#/${path.replace(/^\//, '')}`;
  if (window.location.hash === targetHash) {
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  } else {
    window.location.hash = targetHash;
  }
}

export function useRoute(): {
  route: ParsedRoute;
  navigate: (path: string) => void;
} {
  const [route, setRoute] = useState<ParsedRoute>(() => parseHash(window.location.hash));

  useEffect(() => {
    const handleHashChange = () => {
      setRoute(parseHash(window.location.hash));
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const nav = useCallback((path: string) => {
    navigate(path);
  }, []);

  return { route, navigate: nav };
}
