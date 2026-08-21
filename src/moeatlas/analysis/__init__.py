"""Bounded, read-only routing-load analysis over immutable routing shards."""

from .association_stability import (
    ASSOCIATION_STABILITY_SCHEMA_VERSION,
    AssociationStability,
    AssociationStabilityError,
    analyze_association_stability,
)
from .bundle import (
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    AnalysisBundleEntry,
    AnalysisBundleReceipt,
    write_analysis_bundle,
)
from .compare_heatmap import (
    ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION,
    render_routing_load_comparison,
)
from .corouting import (
    COROUTING_SCHEMA_VERSION,
    CoRoutingError,
    CoRoutingGraph,
    ExpertCoRoutingCounts,
    summarize_co_routing,
)
from .evidence_cards import (
    EVIDENCE_CARD_SCHEMA_VERSION,
    EVIDENCE_TIERS,
    BehaviorSection,
    CausalitySection,
    EvidenceCard,
    EvidenceCardError,
    RoutingSection,
    StabilitySection,
    TaskAssociationSection,
)
from .route_churn import (
    ROUTE_CHURN_SCHEMA_VERSION,
    RouteChurnError,
    RouteChurnSequences,
    RouteChurnSummary,
    analyze_route_churn,
)
from .router_margin import (
    ROUTER_MARGIN_SCHEMA_VERSION,
    RouterMarginError,
    RouterMarginSamples,
    RouterMarginSummary,
    analyze_router_margin,
)
from .routing_agreement import (
    ROUTING_AGREEMENT_SCHEMA_VERSION,
    PromptRolloutCounts,
    RoutingAgreement,
    RoutingAgreementError,
    analyze_routing_agreement,
)
from .routing_compare import (
    ROUTING_COMPARE_SCHEMA_VERSION,
    RoutingLoadComparison,
    compare_routing_load,
)
from .routing_heatmap import (
    ROUTING_HEATMAP_SCHEMA_VERSION,
    render_mixtral_routing_load_heatmap,
    render_routing_load_heatmap,
)
from .routing_load import (
    ROUTING_LOAD_SCHEMA_VERSION,
    MixtralRoutingLoadMatrix,
    RoutingLoadError,
    RoutingLoadMatrix,
    aggregate_mixtral_routing_load,
    aggregate_routing_load,
)
from .routing_summary import (
    ROUTING_SUMMARY_SCHEMA_VERSION,
    RoutingLoadSummary,
    summarize_routing_load,
)
from .task_association import (
    TASK_ASSOCIATION_SCHEMA_VERSION,
    TaskAssociationError,
    TaskAssociationMatrix,
    TaskExpertCounts,
    analyze_task_association,
)

__all__ = [
    "ASSOCIATION_STABILITY_SCHEMA_VERSION",
    "AssociationStability",
    "AssociationStabilityError",
    "analyze_association_stability",
    "EVIDENCE_CARD_SCHEMA_VERSION",
    "EVIDENCE_TIERS",
    "COROUTING_SCHEMA_VERSION",
    "CoRoutingError",
    "CoRoutingGraph",
    "ExpertCoRoutingCounts",
    "summarize_co_routing",
    "ROUTING_AGREEMENT_SCHEMA_VERSION",
    "ROUTER_MARGIN_SCHEMA_VERSION",
    "ROUTE_CHURN_SCHEMA_VERSION",
    "RouteChurnError",
    "RouteChurnSequences",
    "RouteChurnSummary",
    "analyze_route_churn",
    "RouterMarginError",
    "RouterMarginSamples",
    "RouterMarginSummary",
    "analyze_router_margin",
    "BehaviorSection",
    "CausalitySection",
    "EvidenceCard",
    "EvidenceCardError",
    "RoutingSection",
    "StabilitySection",
    "TaskAssociationSection",
    "PromptRolloutCounts",
    "RoutingAgreement",
    "RoutingAgreementError",
    "analyze_routing_agreement",
    "TASK_ASSOCIATION_SCHEMA_VERSION",
    "TaskAssociationError",
    "TaskAssociationMatrix",
    "TaskExpertCounts",
    "analyze_task_association",
    "ANALYSIS_BUNDLE_SCHEMA_VERSION",
    "AnalysisBundleEntry",
    "AnalysisBundleReceipt",
    "write_analysis_bundle",
    "ROUTING_COMPARE_HEATMAP_SCHEMA_VERSION",
    "render_routing_load_comparison",
    "ROUTING_COMPARE_SCHEMA_VERSION",
    "RoutingLoadComparison",
    "compare_routing_load",
    "ROUTING_SUMMARY_SCHEMA_VERSION",
    "RoutingLoadSummary",
    "summarize_routing_load",
    "ROUTING_LOAD_SCHEMA_VERSION",
    "RoutingLoadMatrix",
    "MixtralRoutingLoadMatrix",
    "RoutingLoadError",
    "aggregate_routing_load",
    "aggregate_mixtral_routing_load",
    "ROUTING_HEATMAP_SCHEMA_VERSION",
    "render_routing_load_heatmap",
    "render_mixtral_routing_load_heatmap",
]
