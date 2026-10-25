import React, { useEffect, useState } from 'react';
import { X, FileText, Bookmark, Layers, AlertCircle, Loader2, Link2, HelpCircle } from 'lucide-react';
import type { EvidenceDetail } from '../types';
import { api } from '../api/client';
import { Badge } from './common/Badge';
import { getStatementTypeBadgeVariant } from './common/badgeVariants';
import { NotRecorded } from './common/NotRecorded';

interface EvidenceDrawerProps {
  statementId: string | null;
  onClose: () => void;
}

export const EvidenceDrawer: React.FC<EvidenceDrawerProps> = ({ statementId, onClose }) => {
  const [evidence, setEvidence] = useState<EvidenceDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!statementId) {
      setEvidence(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.getEvidenceDetail(statementId)
      .then((data) => setEvidence(data))
      .catch((err: Error) => setError(err.message || 'Failed to load evidence detail'))
      .finally(() => setLoading(false));
  }, [statementId]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!statementId) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="drawer-header">
          <div className="u-row">
            <Bookmark size={20} className="tone-indigo" />
            <h3 className="drawer-title">Statement Source Provenance</h3>
          </div>
          <button
            onClick={onClose}
            className="drawer-close-btn"
            aria-label="Close drawer"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="drawer-body">
          {loading && (
            <div className="drawer-loading">
              <Loader2 className="drawer-loading-spinner animate-spin" />
              <p>Retrieving statement source provenance...</p>
              <p className="u-text-sm u-mono">Statement ID: {statementId}</p>
            </div>
          )}

          {error && (
            <div className="error-banner">
              <div className="error-banner-content">
                <AlertCircle className="error-banner-icon" />
                <div>
                  <h4 className="error-banner-title">Unable to fetch evidence detail</h4>
                  <p className="error-banner-message">{error}</p>
                </div>
              </div>
            </div>
          )}

          {evidence && !loading && (
            <div className="u-stack-lg">
              {/* Paper Source */}
              <div className="provenance-card">
                <div className="provenance-meta-tag">
                  <FileText size={16} />
                  <span>Source Paper</span>
                </div>
                <h4 className="provenance-paper-title">
                  {evidence.paper_title || `Paper ID: ${evidence.paper_id}`}
                </h4>
                <div className="provenance-paper-id">
                  Paper UUID: {evidence.paper_id}
                </div>
              </div>

              {/* Location Provenance */}
              <div className="grid-2-cols grid-compact">
                <div className="provenance-meta-box">
                  <span className="meta-box-label">Paper Section</span>
                  <span className="meta-box-value">
                    {evidence.section || <NotRecorded />}
                  </span>
                </div>
                <div className="provenance-meta-box">
                  <span className="meta-box-label">Page Range</span>
                  <span className="meta-box-value">
                    {evidence.page_start !== null && evidence.page_start !== undefined
                      ? `Page ${evidence.page_start}${evidence.page_end && evidence.page_end !== evidence.page_start ? `–${evidence.page_end}` : ''}`
                      : <NotRecorded />}
                  </span>
                </div>
              </div>

              {/* Extracted Research Statement */}
              <div className="statement-provenance-box">
                <div className="statement-header-row">
                  <div className="u-row">
                    <span className="label-caps">Extracted Statement</span>
                    <Badge label="Model-Extracted" variant="model-generated" />
                  </div>
                  <Badge
                    label={evidence.statement_type}
                    variant={getStatementTypeBadgeVariant(evidence.statement_type)}
                  />
                </div>
                <div className="statement-text-highlight">
                  "{evidence.statement_text}"
                </div>
              </div>

              {/* Linked Source Passage (this endpoint returns the linked passage; it does not verify it) */}
              <div className="passage-provenance-box">
                <div className="passage-header-row">
                  <div className="u-row label-caps">
                    <Layers size={16} />
                    <span>Source Passage</span>
                  </div>
                  {evidence.passage_text ? (
                    <span className="provenance-status-pill">
                      <Link2 size={14} />
                      <span>Linked source passage</span>
                    </span>
                  ) : (
                    <span className="provenance-status-pill is-missing">
                      <HelpCircle size={14} />
                      <span>Source passage unavailable</span>
                    </span>
                  )}
                </div>
                {evidence.passage_text ? (
                  <div className="passage-text-block">
                    {evidence.passage_text}
                  </div>
                ) : (
                  <div className="passage-unavailable-note">
                    No passage excerpt was linked to this statement ID in the database.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="drawer-footer">
          <button
            onClick={onClose}
            className="btn btn-secondary"
          >
            Close Provenance Panel
          </button>
        </div>
      </div>
    </div>
  );
};
