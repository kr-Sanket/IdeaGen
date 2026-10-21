import { useState, useEffect, useCallback } from 'react';

export type RouteName = 'overview' | 'opportunities' | 'opportunity-detail' | 'corpus';

export interface ParsedRoute {
  name: RouteName;
  params: Record<string, string>;
  rawHash: string;
}

export function parseHash(hash: string): ParsedRoute {
  const clean = hash.replace(/^#\/?/, '');
  const [pathPart, queryPart] = clean.split('?');
  const segments = pathPart ? pathPart.split('/') : [];
  const searchParams = new URLSearchParams(queryPart || '');
  const params: Record<string, string> = {};

  searchParams.forEach((val, key) => {
    params[key] = val;
  });

  if (segments.length === 0 || segments[0] === 'overview' || segments[0] === '') {
    return { name: 'overview', params, rawHash: hash };
  }

  if (segments[0] === 'opportunities') {
    if (segments.length > 1 && segments[1]) {
      return {
        name: 'opportunity-detail',
        params: { ...params, id: segments[1] },
        rawHash: hash,
      };
    }
    return { name: 'opportunities', params, rawHash: hash };
  }

  if (segments[0] === 'corpus' || segments[0] === 'papers') {
    if (segments[0] === 'papers' && segments.length > 1 && segments[1]) {
      params['paper_id'] = segments[1];
    }
    return { name: 'corpus', params, rawHash: hash };
  }

  return { name: 'overview', params, rawHash: hash };
}

export function navigate(path: string): void {
  const targetHash = path.startsWith('#') ? path : `#/${path.replace(/^\//, '')}`;
  if (window.location.hash === targetHash) {
    // Force trigger if same hash to re-render if needed
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
