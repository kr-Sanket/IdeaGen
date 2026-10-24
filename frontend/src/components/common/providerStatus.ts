import type { SystemStatus } from '../../types';

export type ProviderHealthVariant = 'success' | 'warning' | 'danger' | 'default' | 'info';

export interface ProviderHealthDescriptor {
  label: string;
  variant: ProviderHealthVariant;
  ok: boolean;
  isIdle: boolean;
  tooltip?: string;
}

/**
 * Shared provider health interpretation used across Project Overview, Settings, and Dashboard.
 * Distinguishes online/ready, starting, managed idle (normal on-demand state), and real connection failure.
 */
export function describeProviderHealth(status: SystemStatus | null): ProviderHealthDescriptor {
  if (!status) {
    return {
      label: 'Checking status...',
      variant: 'default',
      ok: false,
      isIdle: false,
      tooltip: 'System provider status is loading.',
    };
  }

  if (!status.database.connected) {
    return {
      label: 'Database error',
      variant: 'danger',
      ok: false,
      isIdle: false,
      tooltip: 'Could not establish connection to the configured database.',
    };
  }

  const ollama = status.ollama;

  // External / non-Ollama provider (e.g. Gemini)
  if (!ollama && status.llm.provider !== 'ollama' && status.embedding.provider !== 'ollama') {
    if (status.llm.api_key_configured === false) {
      return {
        label: `${status.llm.provider.toUpperCase()} API key not set`,
        variant: 'warning',
        ok: false,
        isIdle: false,
        tooltip: 'An API key must be configured in the environment.',
      };
    }
    return {
      label: 'Database and models ready',
      variant: 'success',
      ok: true,
      isIdle: false,
      tooltip: 'Configured API providers and database are available.',
    };
  }

  if (ollama) {
    // 1. Ollama is reachable and responding to probes
    if (ollama.reachable) {
      if (ollama.missing_models && ollama.missing_models.length > 0) {
        return {
          label: `Missing model: ${ollama.missing_models.join(', ')}`,
          variant: 'warning',
          ok: false,
          isIdle: false,
          tooltip: `Ollama is reachable, but required model(s) are not downloaded: ${ollama.missing_models.join(', ')}. Run "ollama pull <model>".`,
        };
      }
      return {
        label: 'Database and models ready',
        variant: 'success',
        ok: true,
        isIdle: false,
        tooltip: 'Ollama is online and configured models are installed and reachable.',
      };
    }

    // 2. Ollama is currently unreachable
    if (ollama.auto_manage) {
      // Subprocess is starting
      if (ollama.process_running) {
        return {
          label: 'Starting Ollama...',
          variant: 'info',
          ok: false,
          isIdle: false,
          tooltip: 'The managed Ollama daemon has been launched and is initializing.',
        };
      }

      // Normal managed idle state
      return {
        label: 'Ollama idle (managed on-demand)',
        variant: 'default',
        ok: true,
        isIdle: true,
        tooltip: `Ollama is managed by IdeaGen and starts automatically on-demand when a research run begins (idle timeout: ${ollama.idle_timeout ?? 60}s).`,
      };
    }

    // Unmanaged external endpoint is unreachable: confirmed error
    return {
      label: 'Ollama unreachable',
      variant: 'danger',
      ok: false,
      isIdle: false,
      tooltip: ollama.error || 'Cannot reach configured Ollama endpoint.',
    };
  }

  return {
    label: 'Status unavailable',
    variant: 'default',
    ok: false,
    isIdle: false,
  };
}

export type ModelStatusState = 'installed' | 'missing' | 'idle_configured' | 'unavailable' | 'checking';

export interface ModelStatusDescriptor {
  state: ModelStatusState;
  badgeLabel: string;
  badgeVariant: 'success' | 'danger' | 'default' | 'info' | 'warning';
  tooltip: string;
}

/**
 * Describes the availability of an individual configured model.
 * Does not falsely claim a model is unavailable when the managed daemon is simply idle.
 */
export function describeModelStatus(
  status: SystemStatus | null,
  modelName: string | null,
  provider: string
): ModelStatusDescriptor {
  if (!status || !modelName) {
    return {
      state: 'checking',
      badgeLabel: 'Checking...',
      badgeVariant: 'default',
      tooltip: 'Model status is loading.',
    };
  }

  if (provider === 'gemini') {
    const hasKey = Boolean(status.llm.api_key_configured);
    return {
      state: hasKey ? 'installed' : 'unavailable',
      badgeLabel: hasKey ? 'API key set' : 'No API key',
      badgeVariant: hasKey ? 'success' : 'danger',
      tooltip: hasKey ? 'Gemini API key is configured.' : 'Gemini API key is missing.',
    };
  }

  if (provider === 'ollama') {
    const ollama = status.ollama;
    if (!ollama) {
      return {
        state: 'checking',
        badgeLabel: 'Checking...',
        badgeVariant: 'default',
        tooltip: 'Ollama status is loading.',
      };
    }

    if (ollama.reachable) {
      const isMissing = ollama.missing_models && ollama.missing_models.includes(modelName);
      if (isMissing) {
        return {
          state: 'missing',
          badgeLabel: 'Missing',
          badgeVariant: 'danger',
          tooltip: `Model "${modelName}" is not installed in Ollama. Pull it with "ollama pull ${modelName}".`,
        };
      }
      return {
        state: 'installed',
        badgeLabel: 'Installed',
        badgeVariant: 'success',
        tooltip: `Model "${modelName}" is installed and available in Ollama.`,
      };
    }

    // Auto-managed idle state: model availability is verified when daemon boots
    if (ollama.auto_manage && !ollama.process_running) {
      return {
        state: 'idle_configured',
        badgeLabel: 'Configured (idle)',
        badgeVariant: 'default',
        tooltip: `Model "${modelName}" is configured. Availability will be verified when Ollama starts on demand.`,
      };
    }

    if (ollama.auto_manage && ollama.process_running) {
      return {
        state: 'checking',
        badgeLabel: 'Starting...',
        badgeVariant: 'info',
        tooltip: 'Ollama is launching; checking model availability momentarily.',
      };
    }

    return {
      state: 'unavailable',
      badgeLabel: 'Unavailable',
      badgeVariant: 'danger',
      tooltip: ollama.error || `Cannot reach Ollama endpoint to verify model "${modelName}".`,
    };
  }

  return {
    state: 'checking',
    badgeLabel: 'Configured',
    badgeVariant: 'default',
    tooltip: `Configured under provider "${provider}".`,
  };
}

/**
 * Describes the Ollama HTTP endpoint status truthfully.
 */
export function describeEndpointStatus(status: SystemStatus | null): {
  badgeLabel: string;
  badgeVariant: 'success' | 'danger' | 'default' | 'info';
  isIdle: boolean;
  tooltip: string;
} {
  const ollama = status?.ollama;
  if (!ollama) {
    return {
      badgeLabel: 'Checking...',
      badgeVariant: 'default',
      isIdle: false,
      tooltip: 'Ollama endpoint probe is loading.',
    };
  }

  if (ollama.reachable) {
    return {
      badgeLabel: 'Reachable',
      badgeVariant: 'success',
      isIdle: false,
      tooltip: 'Ollama HTTP daemon is reachable and responding.',
    };
  }

  if (ollama.auto_manage) {
    if (ollama.process_running) {
      return {
        badgeLabel: 'Starting...',
        badgeVariant: 'info',
        isIdle: false,
        tooltip: 'Ollama process has been started and is opening its port.',
      };
    }
    return {
      badgeLabel: 'Idle (on-demand)',
      badgeVariant: 'default',
      isIdle: true,
      tooltip: `Ollama is auto-managed by IdeaGen and will start automatically when needed (idle timeout: ${ollama.idle_timeout ?? 60}s).`,
    };
  }

  return {
    badgeLabel: 'Unreachable',
    badgeVariant: 'danger',
    isIdle: false,
    tooltip: ollama.error || 'Could not connect to configured Ollama endpoint.',
  };
}
