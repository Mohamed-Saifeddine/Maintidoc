from maintidoc.models import ElementType, SearchHit
from maintidoc.retrieval import reciprocal_rank_fusion


def hit(element_id: str, score: float = 0.5) -> SearchHit:
    return SearchHit(
        document_id="doc",
        file_name="report.pdf",
        page=1,
        element_id=element_id,
        element_type=ElementType.TEXT,
        content="contenu",
        score=score,
    )


def test_rrf_prioritises_hit_present_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("b"), hit("c")]])
    assert fused[0].element_id == "b"
    assert len(fused) == 3
