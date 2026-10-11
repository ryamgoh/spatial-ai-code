import pytest

from spatial.v2.hitl.batches import aggregate_review_batches, batch_review_packet
from spatial.v2.hitl.review import (
    FinalDisposition,
    ReviewItem,
    ReviewPacket,
    ReviewRouting,
    RoutedReviewItem,
)


def test_bounded_batches_cover_parent_once_and_reject_missing_batch():
    parent = ReviewPacket.create(
        packet_id="p",
        registry_fingerprint="r",
        rubric_version="v",
        items=tuple(ReviewItem.create(str(i), "prompt", "trace") for i in range(5)),
    )
    batches = batch_review_packet(parent, max_items=2)
    assert [len(b.items) for b in batches] == [2, 2, 1]
    routed = tuple(
        ReviewRouting(
            b.packet_id,
            b.fingerprint,
            "d",
            "g",
            tuple(RoutedReviewItem(i, FinalDisposition.ACCEPT) for i in b.items),
        )
        for b in batches
    )
    assert aggregate_review_batches(parent, batches, routed)["counts"] == {"accept": 5}
    with pytest.raises(ValueError, match="cover"):
        aggregate_review_batches(parent, batches[:-1], routed[:-1])
