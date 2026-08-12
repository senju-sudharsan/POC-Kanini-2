export type AuraState =
  | 'idle'
  | 'focus'
  | 'typing'
  | 'processing'
  | 'retrieval'
  | 'vision'
  | 'ml'
  | 'result'
  | 'hitl'
  | 'error';

export const STATE_LABELS: Record<AuraState, string> = {
  idle: 'Ready',
  focus: 'Listening',
  typing: 'Composing',
  processing: 'Reasoning',
  retrieval: 'Searching Evidence',
  vision: 'Analyzing Visuals',
  ml: 'Computing Model',
  result: 'Complete',
  hitl: 'Approval Required',
  error: 'Attention Needed',
};

export function AuraOrb({ state }: { state: AuraState }) {
  return (
    <div className="orb-stage" data-state={state} aria-hidden="true">
      <div className="orb-aura" />
      <div className="orb-shell">
        <div className="orb-core" />
        <span className="orb-particle orb-particle-one" />
        <span className="orb-particle orb-particle-two" />
        <span className="orb-particle orb-particle-three" />
      </div>
    </div>
  );
}
